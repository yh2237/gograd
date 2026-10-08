package irodori

import (
	"fmt"
	"math"
	"strings"
)

var annotationEmojis = []string{
	"⏩", "⏱️", "⏸️", "🌬️", "🍭", "🎛️", "🎭", "🎵", "🐢", "🐱", "👂", "👃", "👅", "👌", "👏", "💋", "💥", "💦", "💪", "📄", "📞", "📢", "📣", "😆", "😊", "😌", "😎", "😏", "😒", "😖", "😟", "😠", "😪", "😭", "😮‍💨", "😮", "😰", "😱", "😲", "😴", "🙄", "🙏", "🤐", "🤔", "🤢", "🤧", "🤭", "🥤", "🥱", "🥴", "🥵", "🥹", "🥺", "🫣", "🫶", "📖",
}

func countEmoji(text string) int {
	count := 0
	for len(text) > 0 {
		matched := false
		for _, emoji := range annotationEmojis {
			if strings.HasPrefix(text, emoji) {
				text = text[len(emoji):]
				count++
				matched = true
				break
			}
		}
		if !matched {
			_, width := firstRune(text)
			text = text[width:]
		}
	}
	return count
}

// firstRune avoids changing byte-index semantics when scanning annotations.
func firstRune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}

func logCap(value, cap int) float32 {
	return float32(math.Log1p(float64(min(max(value, 0), cap))) / math.Log1p(float64(cap)))
}

// DurationFeatures reproduces the reference's fourteen auxiliary features.
// text must already have passed Irodori's text normalization.
func DurationFeatures(text string, tokenCount, maxTextLen int, hasSpeaker bool) ([14]float32, error) {
	var out [14]float32
	if maxTextLen <= 0 {
		return out, fmt.Errorf("irodori: max text length must be positive")
	}
	charCount := max(len([]rune(text)), 1)
	var kana, kanji, alnum int
	for _, r := range text {
		switch {
		case r >= 0x3040 && r <= 0x309f || r >= 0x30a0 && r <= 0x30ff:
			kana++
		case r >= 0x3400 && r <= 0x4dbf || r >= 0x4e00 && r <= 0x9fff || r >= 0xf900 && r <= 0xfaff || r >= 0x20000 && r <= 0x2fa1f:
			kanji++
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9':
			alnum++
		}
	}
	out[0] = float32(min(max(tokenCount, 0), maxTextLen)) / float32(maxTextLen)
	out[1] = logCap(charCount, 512)
	out[2] = float32(tokenCount) / float32(charCount)
	out[3] = logCap(strings.Count(text, "。")+strings.Count(text, "."), 8)
	out[4] = logCap(strings.Count(text, "、")+strings.Count(text, ","), 16)
	out[5] = logCap(strings.Count(text, "ー"), 8)
	out[6] = logCap(strings.Count(text, "…"), 8)
	out[7] = logCap(strings.Count(text, "！")+strings.Count(text, "!"), 8)
	out[8] = logCap(strings.Count(text, "？")+strings.Count(text, "?"), 8)
	out[9] = logCap(countEmoji(text), 8)
	out[10] = float32(kana) / float32(charCount)
	out[11] = float32(kanji) / float32(charCount)
	out[12] = float32(alnum) / float32(charCount)
	if hasSpeaker {
		out[13] = 1
	}
	return out, nil
}
