package autograd

import (
	"maps"
	"slices"
	"testing"
)

func TestRegisteredElementwiseOpsMatchesTable(t *testing.T) {
	got := RegisteredElementwiseOps()
	want := slices.Sorted(maps.Keys(elementwiseRegistry))
	if !slices.Equal(got, want) {
		t.Fatalf("RegisteredElementwiseOps() = %v, want %v", got, want)
	}
	if !slices.IsSorted(got) {
		t.Fatalf("RegisteredElementwiseOps() = %v, want sorted output", got)
	}
	for _, name := range got {
		if _, ok := operationRegistry[name]; !ok {
			t.Fatalf("%q is in elementwiseRegistry but not operationRegistry", name)
		}
	}
}
