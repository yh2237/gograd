package tensor

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
)

// ZerosOn must return a zero-filled tensor on the requested device and reject
// invalid shapes before allocating.
func TestZerosOn(t *testing.T) {
	for _, device := range []Device{CPU, CUDA} {
		if device == CUDA && !cuda.Available() {
			continue
		}
		t.Run(string(device), func(t *testing.T) {
			tr, err := ZerosOn(device, 2, 3)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			if tr.Device() != device {
				t.Fatalf("device %s, want %s", tr.Device(), device)
			}
			if shape := tr.Shape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
				t.Fatalf("shape %v, want [2 3]", shape)
			}
			host, err := tr.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			if len(host) != 6 {
				t.Fatalf("host length %d, want 6", len(host))
			}
			for i, v := range host {
				if v != 0 {
					t.Fatalf("element %d = %g, want 0", i, v)
				}
			}
			if _, err := ZerosOn(device, -1); err == nil {
				t.Fatal("ZerosOn accepted a negative dimension")
			}
		})
	}
}
