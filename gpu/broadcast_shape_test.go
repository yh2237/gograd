package gpu

import (
	"slices"
	"testing"
)

// BroadcastShape right-aligns the operands, expands size-1 axes and rejects
// shapes that cannot be broadcast.
func TestBroadcastShape(t *testing.T) {
	ok := []struct {
		a, b, want []int
	}{
		{[]int{3}, []int{3}, []int{3}},
		{[]int{3}, []int{2, 3}, []int{2, 3}},
		{[]int{1, 3}, []int{2, 1}, []int{2, 3}},
		{[]int{2, 1, 4}, []int{3, 1}, []int{2, 3, 4}},
		{nil, []int{2, 3}, []int{2, 3}},
		{nil, nil, []int{}},
	}
	for _, tc := range ok {
		got, err := BroadcastShape(tc.a, tc.b)
		if err != nil {
			t.Fatalf("BroadcastShape(%v, %v): %v", tc.a, tc.b, err)
		}
		if !slices.Equal(got, tc.want) {
			t.Fatalf("BroadcastShape(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}

	bad := [][2][]int{
		{{2, 3}, {3, 2}},
		{{2}, {3}},
		{{1, 2, 3, 4, 5, 6, 7}, {1}},
	}
	for _, tc := range bad {
		if _, err := BroadcastShape(tc[0], tc[1]); err == nil {
			t.Fatalf("BroadcastShape(%v, %v) accepted incompatible shapes", tc[0], tc[1])
		}
	}
}
