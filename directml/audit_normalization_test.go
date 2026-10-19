package directml

import (
	"math"
	"testing"
)

// TestAuditDirectMLNormalizationSingleElementGroup covers the degenerate
// normalization: a group with a single element normalizes to zero, so the
// result is the bias alone. Some DirectML implementations report a NaN
// variance for a one-element group, which would poison the whole tensor.
func TestAuditDirectMLNormalizationSingleElementGroup(t *testing.T) {
	b := testBackend(t)
	check := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: length %d != %d", name, len(got), len(want))
		}
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 1e-4 {
				t.Fatalf("%s[%d] = %v, want %v", name, i, v, want[i])
			}
		}
	}
	t.Run("layernorm", func(t *testing.T) {
		for _, value := range []float32{3, -2.125, 0, -1e4} {
			data := []float32{value, value, value}
			bias := []float32{-0.625}
			out := alloc(t, b, len(data))
			if err := b.LayerNorm(out, upload(t, b, data), upload(t, b, []float32{1.5}), upload(t, b, bias), 3, 1, 1e-5); err != nil {
				t.Fatal(err)
			}
			got, err := b.Download(out, len(data))
			if err != nil {
				t.Fatal(err)
			}
			check("layernorm", got, []float32{-0.625, -0.625, -0.625})
		}
	})
	t.Run("groupnorm", func(t *testing.T) {
		// One frame and one channel per group: every group holds a single
		// element and normalizes to zero.
		const batch, frames, channels, groups = 2, 1, 3, 3
		data := make([]float32, batch*frames*channels)
		for i := range data {
			data[i] = float32(i%7-3) * 1.5
		}
		bias := []float32{-0.5, 0.25, 1}
		out := alloc(t, b, len(data))
		if err := b.GroupNorm(out, upload(t, b, data), upload(t, b, []float32{1, 1, 1}), upload(t, b, bias), batch, frames, channels, groups, 1e-5); err != nil {
			t.Fatal(err)
		}
		got, err := b.Download(out, len(data))
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float32, len(data))
		for i := range want {
			want[i] = bias[i%channels]
		}
		check("groupnorm", got, want)
	})
}
