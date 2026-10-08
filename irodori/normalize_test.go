package irodori

import (
	"encoding/json"
	"os"
	"testing"
)

func TestNormalizeTextPyTorchFixture(t *testing.T) {
	data, err := os.ReadFile("../testdata/irodori_text.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Input  string `json:"input"`
		Output string `json:"output"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if got := NormalizeText(c.Input); got != c.Output {
			t.Errorf("NormalizeText(%q) = %q, want %q", c.Input, got, c.Output)
		}
	}
}
