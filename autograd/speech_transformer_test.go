package autograd

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestSpeechTransformerPyTorchForward(t *testing.T) {
	b, err := os.ReadFile("../testdata/speech_transformer.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		RotaryInput     []float32 `json:"rotary_input"`
		RotaryOutput    []float32 `json:"rotary_output"`
		GeGLUInput      []float32 `json:"geglu_input"`
		GeGLUOutput     []float32 `json:"geglu_output"`
		SwiGLUInput     []float32 `json:"swiglu_input"`
		W1              []float32 `json:"w1"`
		W2              []float32 `json:"w2"`
		W3              []float32 `json:"w3"`
		SwiGLUOutput    []float32 `json:"swiglu_output"`
		AttentionQ      []float32 `json:"attention_q"`
		AttentionK      []float32 `json:"attention_k"`
		AttentionV      []float32 `json:"attention_v"`
		AttentionOutput []float32 `json:"attention_output"`
		EmbeddingWeight []float32 `json:"embedding_weight"`
		EmbeddingIDs    []int     `json:"embedding_ids"`
		EmbeddingOutput []float32 `json:"embedding_output"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		name := "cpu"
		if device == tensor.CUDA {
			name = "cuda"
		}
		t.Run(name, func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			if device == tensor.CUDA {
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			makeTensor := func(v []float32, shape ...int) *Tensor {
				z, e := New(v, shape, device, false)
				if e != nil {
					t.Fatal(e)
				}
				return z
			}
			check := func(label string, y *Tensor, want []float32) {
				got, e := y.ToHost()
				if e != nil {
					t.Fatal(e)
				}
				if len(got) != len(want) {
					t.Fatalf("%s length", label)
				}
				var maxAbs float64
				for i, v := range got {
					maxAbs = max(maxAbs, math.Abs(float64(v-want[i])))
				}
				t.Logf("%s max_abs=%.9g", label, maxAbs)
				if maxAbs > 5e-5 {
					t.Errorf("%s mismatch", label)
				}
				y.Close()
			}
			x := makeTensor(f.RotaryInput, 1, 2, 4, 8)
			defer x.Close()
			check("RotaryHalf", RotaryHalf(x, 160000), f.RotaryOutput)
			ge := makeTensor(f.GeGLUInput, 2, 12)
			defer ge.Close()
			check("GeGLU", GeGLU(ge), f.GeGLUOutput)
			sx := makeTensor(f.SwiGLUInput, 2, 4)
			defer sx.Close()
			w1 := makeTensor(f.W1, 6, 4)
			defer w1.Close()
			w2 := makeTensor(f.W2, 4, 6)
			defer w2.Close()
			w3 := makeTensor(f.W3, 6, 4)
			defer w3.Close()
			check("SwiGLUProjection", SwiGLUProjection(sx, w1, w2, w3), f.SwiGLUOutput)
			q := makeTensor(f.AttentionQ, 1, 2, 4, 8)
			defer q.Close()
			k := makeTensor(f.AttentionK, 1, 2, 4, 8)
			defer k.Close()
			v := makeTensor(f.AttentionV, 1, 2, 4, 8)
			defer v.Close()
			check("KeyPaddingAttention", KeyPaddingAttention(q, k, v, []bool{true, true, false, true}, 1), f.AttentionOutput)
			ew := makeTensor(f.EmbeddingWeight, 9, 4)
			defer ew.Close()
			check("Embedding", Embedding(ew, f.EmbeddingIDs, []int{4}), f.EmbeddingOutput)
		})
	}
}
