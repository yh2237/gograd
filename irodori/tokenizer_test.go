package irodori

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCachedTokenizerParity(t *testing.T) {
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for cached tokenizer parity")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "tokenizer", "tokenizer.json"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("tokenizer path: %v %v", paths, err)
	}
	tok, err := LoadTokenizer(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../testdata/irodori_tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text      string `json:"text"`
		IDs       []int  `json:"ids"`
		Padded1   []int  `json:"padded_1"`
		Padded8   []int  `json:"padded_8"`
		Padded32  []int  `json:"padded_32"`
		Padded256 []int  `json:"padded_256"`
	}
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		got := append([]int{1}, tok.Encode(c.Text)...)
		if !reflect.DeepEqual(got, c.IDs) {
			t.Errorf("case %d body: got %v want %v", i, got, c.IDs)
		}
		for _, p := range []struct {
			n   int
			ids []int
		}{{1, c.Padded1}, {8, c.Padded8}, {32, c.Padded32}, {256, c.Padded256}} {
			batch, _, err := tok.BatchEncode([]string{c.Text}, p.n)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(batch[0], p.ids) {
				t.Errorf("case %d padded %d: got %v want %v", i, p.n, batch[0], p.ids)
			}
		}
	}
	t.Logf("matched %d cached-tokenizer cases", len(cases))
}
