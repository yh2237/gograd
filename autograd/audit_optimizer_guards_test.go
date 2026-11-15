package autograd

import (
	"math"
	"path/filepath"
	"testing"
)

func auditParam(t *testing.T, name string, values ...float32) Parameter {
	t.Helper()
	v := Must(values, []int{len(values)}, true)
	t.Cleanup(v.Close)
	return Parameter{Name: name, Value: v}
}

// A negative or non-finite learning rate would turn the update into ascent.
func TestAuditAdamWRejectsInvalidLearningRate(t *testing.T) {
	for _, lr := range []float32{-0.1, float32(math.NaN()), float32(math.Inf(1))} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("NewAdamW accepted learning rate %v", lr)
				}
			}()
			NewAdamW([]Parameter{auditParam(t, "w", 1)}, lr, 0)
		}()
	}
}

// State is keyed by name, so duplicates would silently swap moments.
func TestAuditAdamWRejectsDuplicateParameterNames(t *testing.T) {
	params := []Parameter{auditParam(t, "w", 1), auditParam(t, "w", 2)}
	if _, err := NewAdamWGroups(params, .1, []ParameterGroupSpec{{Name: "default", Params: params}}); err == nil {
		t.Fatal("NewAdamWGroups accepted duplicate parameter names")
	}
}

// A non-positive peak would pin the schedule at zero or drive it negative.
func TestAuditOneCycleRejectsNonPositivePeak(t *testing.T) {
	for _, maxLR := range []float64{0, -1} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("NewOneCycle accepted peak %v", maxLR)
				}
			}()
			NewOneCycle(maxLR, 10, .3)
		}()
	}
}

// Parameters and buffers share one namespace, so a collision must be reported
// rather than dropping a tensor from the checkpoint.
func TestAuditSafeTensorsRejectsDuplicateStateName(t *testing.T) {
	m := &Module{
		Parameters: []Parameter{auditParam(t, "x", 1, 2)},
		Buffers:    []Parameter{auditParam(t, "x", 9, 9)},
	}
	path := filepath.Join(t.TempDir(), "dup.safetensors")
	if err := m.SaveSafeTensors(path); err == nil {
		t.Fatal("SaveSafeTensors accepted a duplicate state name")
	}
}
