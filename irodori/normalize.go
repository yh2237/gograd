package irodori

import (
	"regexp"
	"strings"

	"golang.org/x/text/unicode/norm"
)

var longEllipsis = regexp.MustCompile(`…{3,}`)

var simpleReplacements = strings.NewReplacer(
	"\t", "", "[n]", "", `\[n\]`, "", "　", "",
	"？", "?", "！", "!", "♥", "♡", "●", "○", "◯", "○", "〇", "○",
)

var removedMarks = map[rune]bool{}

func init() {
	for _, r := range ";▼♀♂《》≪≫①②③④⑤⑥˗‐‑‒–—―⁃−⎯⏤─━⸺⸻" {
		removedMarks[r] = true
	}
}

// NormalizeText follows the reference's ordered substitutions and NFKC pass.
// It intentionally preserves ordinary leading and trailing whitespace.
func NormalizeText(input string) string {
	input = simpleReplacements.Replace(input)
	input = strings.Map(func(r rune) rune {
		if removedMarks[r] {
			return -1
		}
		if r == '～' || r == '〜' {
			return 'ー'
		}
		return r
	}, input)
	input = longEllipsis.ReplaceAllString(input, "……")
	input = stripOuterBrackets(input)
	input = norm.NFKC.String(input)
	input = strings.ReplaceAll(input, "...", "…")
	return strings.ReplaceAll(input, "..", "…")
}

func stripOuterBrackets(s string) string {
	pairs := map[rune]rune{'「': '」', '『': '』', '（': '）', '【': '】', '(': ')'}
	for {
		r := []rune(s)
		if len(r) < 2 || pairs[r[0]] != r[len(r)-1] {
			return s
		}
		depth, enclosing := 0, true
		for i, c := range r {
			if c == r[0] {
				depth++
			} else if c == r[len(r)-1] {
				depth--
			}
			if depth == 0 && i < len(r)-1 {
				enclosing = false
				break
			}
		}
		if !enclosing || depth != 0 {
			return s
		}
		s = string(r[1 : len(r)-1])
	}
}
