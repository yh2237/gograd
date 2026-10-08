package irodori

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
)

func TestModernBERTRealCheckpointParity(t *testing.T) {
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for real checkpoint parity")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("checkpoint path: %v %v", paths, err)
	}
	state, err := autograd.OpenSafeTensorFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	b, err := os.ReadFile("../testdata/irodori_modernbert.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		IDs           []int       `json:"ids"`
		Mask          []bool      `json:"mask"`
		Rows          []int       `json:"rows"`
		Slices        [][]float32 `json:"slices"`
		TextSlices    [][]float32 `json:"text_slices"`
		CaptionSlices [][]float32 `json:"caption_slices"`
		RowSums       []float32   `json:"row_sums"`
	}
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	for index, c := range cases {
		start := time.Now()
		got, err := (ModernBERT{State: state}).Forward(c.IDs, c.Mask)
		if err != nil {
			t.Fatal(err)
		}
		var maxSlice, maxSum float64
		for r, row := range c.Rows {
			for j, want := range c.Slices[r] {
				maxSlice = max(maxSlice, math.Abs(float64(got[row*768+j]-want)))
			}
		}
		for row, want := range c.RowSums {
			var sum float32
			for _, v := range got[row*768 : (row+1)*768] {
				sum += v
			}
			maxSum = max(maxSum, math.Abs(float64(sum-want)))
		}
		for _, kind := range []struct {
			name string
			want [][]float32
		}{{"text", c.TextSlices}, {"caption", c.CaptionSlices}} {
			projected, e := (ModernBERT{State: state}).Project(got, c.Mask, kind.name)
			if e != nil {
				t.Fatal(e)
			}
			var maxProject float64
			for r, row := range c.Rows {
				for j, want := range kind.want[r] {
					maxProject = max(maxProject, math.Abs(float64(projected[row*512+j]-want)))
				}
			}
			t.Logf("case %d %s_project_max_abs=%.8g", index, kind.name, maxProject)
			if maxProject > 0.005 {
				t.Errorf("%s projection mismatch", kind.name)
			}
		}
		t.Logf("case %d len=%d max_slice_abs=%.8g max_row_sum_abs=%.8g go_ms=%.3f", index, len(c.IDs), maxSlice, maxSum, float64(time.Since(start).Microseconds())/1000)
		if maxSlice > 0.005 || maxSum > 0.2 {
			t.Errorf("ModernBERT parity exceeds tolerance")
		}
	}
}
