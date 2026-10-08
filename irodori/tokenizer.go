package irodori

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

type unigramToken struct {
	ID    int
	Score float64
}

type tokenNode struct {
	Next  map[byte]*tokenNode
	Token *unigramToken
}

// Tokenizer reads the Unigram/Metaspace tokenizer.json packaged with
// ModernBERT-ja. The vocabulary stays in host memory; checkpoint weights do not.
type Tokenizer struct {
	root         tokenNode
	byteIDs      [256]int
	unknownScore float64
}

func LoadTokenizer(path string) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Model struct {
			Type         string              `json:"type"`
			Vocab        [][]json.RawMessage `json:"vocab"`
			UnkID        int                 `json:"unk_id"`
			ByteFallback bool                `json:"byte_fallback"`
		} `json:"model"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Replacement   string `json:"replacement"`
			PrependScheme string `json:"prepend_scheme"`
			Split         bool   `json:"split"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	if raw.Model.Type != "Unigram" || !raw.Model.ByteFallback || raw.Model.UnkID != 0 || raw.PreTokenizer.Type != "Metaspace" || raw.PreTokenizer.Replacement != "▁" || raw.PreTokenizer.PrependScheme != "never" || raw.PreTokenizer.Split {
		return nil, fmt.Errorf("irodori: unsupported tokenizer configuration")
	}
	t := &Tokenizer{unknownScore: math.Inf(1)}
	for i := range t.byteIDs {
		t.byteIDs[i] = -1
	}
	for id, pair := range raw.Model.Vocab {
		if len(pair) != 2 {
			return nil, fmt.Errorf("irodori: malformed unigram entry %d", id)
		}
		var value string
		var score float64
		if err := json.Unmarshal(pair[0], &value); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, err
		}
		if score < t.unknownScore {
			t.unknownScore = score
		}
		if len(value) == 6 && strings.HasPrefix(value, "<0x") && value[5] == '>' {
			var v byte
			if _, err := fmt.Sscanf(value, "<0x%02X>", &v); err == nil {
				t.byteIDs[v] = id
			}
		}
		node := &t.root
		for _, ch := range []byte(value) {
			if node.Next == nil {
				node.Next = map[byte]*tokenNode{}
			}
			if node.Next[ch] == nil {
				node.Next[ch] = &tokenNode{}
			}
			node = node.Next[ch]
		}
		node.Token = &unigramToken{ID: id, Score: score}
	}
	return t, nil
}

// Encode returns body token IDs without BOS/EOS, matching add_special_tokens=False.
func (t *Tokenizer) Encode(s string) []int {
	s = strings.ReplaceAll(s, " ", "▁")
	n := len(s)
	scores := make([]float64, n+1)
	previous := make([]int, n+1)
	ids := make([]int, n+1)
	for i := 1; i <= n; i++ {
		scores[i] = math.Inf(-1)
	}
	for start := 0; start < n; {
		_, width := utf8.DecodeRuneInString(s[start:])
		if !math.IsInf(scores[start], -1) {
			node := &t.root
			for end := start; end < n; end++ {
				node = node.Next[s[end]]
				if node == nil {
					break
				}
				if node.Token != nil && (end+1 == n || utf8.RuneStart(s[end+1])) {
					candidate := scores[start] + node.Token.Score
					if candidate > scores[end+1] {
						scores[end+1], previous[end+1], ids[end+1] = candidate, start, node.Token.ID
					}
				}
			}
			end := start + width
			candidate := scores[start] + t.unknownScore - 10
			if candidate > scores[end] {
				scores[end], previous[end], ids[end] = candidate, start, 0
			}
		}
		start += width
	}
	var reverse []int
	for at := n; at > 0; at = previous[at] {
		if ids[at] == 0 {
			for j := at - 1; j >= previous[at]; j-- {
				b := s[j]
				id := t.byteIDs[b]
				if id < 0 {
					id = 0
				}
				reverse = append(reverse, id)
			}
		} else {
			reverse = append(reverse, ids[at])
		}
	}
	for i, j := 0, len(reverse)-1; i < j; i, j = i+1, j-1 {
		reverse[i], reverse[j] = reverse[j], reverse[i]
	}
	return reverse
}

// BatchEncode prepends BOS, then truncates and right-pads with the cached
// tokenizer's <pad> ID. The returned mask is true for non-padding positions.
func (t *Tokenizer) BatchEncode(texts []string, maxLength int) ([][]int, [][]bool, error) {
	if len(texts) == 0 || maxLength <= 0 {
		return nil, nil, fmt.Errorf("irodori: invalid tokenizer batch")
	}
	batch, masks := make([][]int, len(texts)), make([][]bool, len(texts))
	for i, s := range texts {
		ids := append([]int{1}, t.Encode(s)...)
		if len(ids) > maxLength {
			ids = ids[:maxLength]
		}
		batch[i], masks[i] = make([]int, maxLength), make([]bool, maxLength)
		for j := range batch[i] {
			batch[i][j] = 3
		}
		copy(batch[i], ids)
		for j := range ids {
			masks[i][j] = true
		}
	}
	return batch, masks, nil
}
