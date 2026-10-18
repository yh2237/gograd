//go:build !windows || !amd64

package directml

import (
	"errors"
	"github.com/yh2237/gograd/gpu"
	"testing"
)

func TestDirectMLUnavailable(t *testing.T) {
	b := NewBackend()
	defer b.Close()
	if b.Available() || !errors.Is(b.Err(), gpu.ErrUnavailable) {
		t.Fatal(b.Err())
	}
	if _, err := b.Alloc(1); !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatal(err)
	}
}
