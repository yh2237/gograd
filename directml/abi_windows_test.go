//go:build windows && amd64

package directml

import "testing"

func TestDirectMLABI(t *testing.T) {
	if err := checkABI(); err != nil {
		t.Fatal(err)
	}
}
