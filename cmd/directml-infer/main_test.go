package main

import (
	"github.com/yh2237/gograd/tensor"
	"testing"
)

func TestCPUInferenceExample(t *testing.T) {
	if err := infer(tensor.CPU, 7, 11, 13); err != nil {
		t.Fatal(err)
	}
}
