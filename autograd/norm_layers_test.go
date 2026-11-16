package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// The layer wrappers must match the function forms and register their state.
func TestAuditNormLayersMatchFunctions(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4, 5, 6}
		rms, err := NewRMSNormLayer(3, 1e-5, device)
		if err != nil {
			t.Fatal(err)
		}
		defer rms.Weight.Close()
		if len(rms.Module.Parameters) != 1 || rms.Module.Parameters[0].Name != "weight" {
			t.Fatalf("RMSNormLayer parameters %v", rms.Module.Parameters)
		}
		x := conv2dTestTensor(t, values, []int{1, 2, 3}, device, true)
		defer x.Close()
		got := conv2dHost(t, rms.Forward(x), false)
		want := conv2dHost(t, RMSNormLast(x, rms.Weight, 1e-5), false)
		if !slices.Equal(got, want) {
			t.Fatalf("RMSNormLayer %v != RMSNormLast %v", got, want)
		}

		group, err := NewGroupNormLayer(3, 1, 1e-5, device)
		if err != nil {
			t.Fatal(err)
		}
		defer group.Weight.Close()
		defer group.Bias.Close()
		if len(group.Module.Parameters) != 2 {
			t.Fatalf("GroupNormLayer parameters %v", group.Module.Parameters)
		}
		got = conv2dHost(t, group.Forward(x), false)
		want = conv2dHost(t, GroupNorm(x, group.Weight, group.Bias, 1, 1e-5), false)
		if !slices.Equal(got, want) {
			t.Fatalf("GroupNormLayer %v != GroupNorm %v", got, want)
		}
	})
}

// Invalid dimensions are rejected at construction.
func TestAuditNormLayersRejectInvalidDimensions(t *testing.T) {
	if _, err := NewRMSNormLayer(0, 1e-5, tensor.CPU); err == nil {
		t.Fatal("NewRMSNormLayer accepted width 0")
	}
	if _, err := NewRMSNormLayer(4, 0, tensor.CPU); err == nil {
		t.Fatal("NewRMSNormLayer accepted epsilon 0")
	}
	if _, err := NewGroupNormLayer(6, 4, 1e-5, tensor.CPU); err == nil {
		t.Fatal("NewGroupNormLayer accepted channels not divisible by groups")
	}
	if _, err := NewGroupNormLayer(6, 0, 1e-5, tensor.CPU); err == nil {
		t.Fatal("NewGroupNormLayer accepted zero groups")
	}
}
