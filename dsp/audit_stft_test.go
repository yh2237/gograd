package dsp

import (
	"testing"
)

func TestAuditSTFTRejectsInvalidHop(t *testing.T) {
	for _, hop := range []int{0, -1, -512, 513} {
		if _, err := NewSTFT(512, hop); err == nil {
			t.Fatalf("NewSTFT(512, %d) was accepted", hop)
		}
	}
	if _, err := NewSTFT(512, 256); err != nil {
		t.Fatalf("NewSTFT(512, 256): %v", err)
	}
}

// TestAuditSTFTFramesForMatchesTransform checks that the reported frame count
// is the one Transform produces.
func TestAuditSTFTFramesForMatchesTransform(t *testing.T) {
	for _, filter := range []int{1, 512} {
		hop := filter / 2
		if hop < 1 {
			hop = 1
		}
		s, err := NewSTFT(filter, hop)
		if err != nil {
			t.Fatal(err)
		}
		for _, length := range []int{1, 1024} {
			want := s.FramesFor(length)
			_, _, frames := s.Transform(make([]float32, length))
			if want != frames {
				t.Fatalf("filter %d length %d: FramesFor=%d, Transform=%d", filter, length, want, frames)
			}
		}
	}
}
