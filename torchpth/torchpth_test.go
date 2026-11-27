package torchpth

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFixture(t *testing.T) {
	expectedBytes, e := os.ReadFile("testdata/tiny.json")
	if e != nil {
		t.Fatal(e)
	}
	var expected map[string]Tensor
	if e = json.Unmarshal(expectedBytes, &expected); e != nil {
		t.Fatal(e)
	}
	got, e := Read("testdata/tiny.pth")
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("got %#v; want %#v", got, expected)
	}
}

func TestRejectGlobal(t *testing.T) {
	name := filepath.Join(t.TempDir(), "invalid.pth")
	f, e := os.Create(name)
	if e != nil {
		t.Fatal(e)
	}
	z := zip.NewWriter(f)
	entry, e := z.Create("invalid/data.pkl")
	if e != nil {
		t.Fatal(e)
	}
	if _, e = entry.Write([]byte("\x80\x02cos\nsystem\n.")); e != nil {
		t.Fatal(e)
	}
	if e = z.Close(); e != nil {
		t.Fatal(e)
	}
	if e = f.Close(); e != nil {
		t.Fatal(e)
	}
	_, e = Read(name)
	if e == nil || !strings.Contains(e.Error(), "forbidden global") {
		t.Fatalf("expected forbidden global, got %v", e)
	}
}

func TestOfficialKokoro(t *testing.T) {
	p := os.Getenv("KOKORO_PTH")
	if p == "" {
		t.Skip("set KOKORO_PTH")
	}
	got, e := Read(p)
	if e != nil {
		t.Fatal(e)
	}
	v, ok := got["bert.module.embeddings.word_embeddings.weight"]
	if !ok {
		t.Fatal("missing word embeddings")
	}
	if !reflect.DeepEqual(v.Shape, []int{178, 128}) {
		t.Errorf("shape %v", v.Shape)
	}
	for i := 0; i < 4; i++ {
		if v.Data[i] != 0 {
			t.Errorf("value %d = %g", i, v.Data[i])
		}
	}
	checks := []struct {
		name        string
		shape       []int
		first, last float32
		index       int
	}{
		{"bert.module.embeddings.position_embeddings.weight", []int{512, 128}, 0.133347749710083, -0.07889200747013092, 123},
		{"bert_encoder.module.weight", []int{512, 768}, 0.10333709418773651, 0.04643180966377258, 123},
		{"predictor.module.duration_proj.linear_layer.bias", []int{50}, 0.635822057723999, -0.41102561354637146, 49},
		{"decoder.module.generator.conv_post.weight_v", []int{22, 128, 7}, -0.06588420271873474, 0.08505526185035706, 123},
		{"text_encoder.module.embedding.weight", []int{178, 512}, 0.1642366349697113, 0.07480001449584961, 123},
	}
	for _, check := range checks {
		v, ok := got[check.name]
		if !ok {
			t.Errorf("missing %s", check.name)
			continue
		}
		if !reflect.DeepEqual(v.Shape, check.shape) || v.Data[0] != check.first || v.Data[check.index] != check.last {
			t.Errorf("%s shape %v samples %g %g", check.name, v.Shape, v.Data[0], v.Data[check.index])
		}
	}
}

func TestOfficialVoice(t *testing.T) {
	p := os.Getenv("KOKORO_VOICE")
	if p == "" {
		t.Skip("set KOKORO_VOICE")
	}
	got, e := Read(p)
	if e != nil {
		t.Fatal(e)
	}
	v, ok := got[""]
	if !ok {
		t.Fatal("missing voice tensor")
	}
	if !reflect.DeepEqual(v.Shape, []int{510, 1, 256}) || v.Data[0] != -0.25429046154022217 || v.Data[123] != 0.18424998223781586 {
		t.Fatalf("voice shape %v first %g sample %g", v.Shape, v.Data[0], v.Data[123])
	}
}

func TestDoubleStorageAndMemoOrderedDict(t *testing.T) {
	// Build a minimal zip with root-level data.pkl containing an OrderedDict
	// stored in memo before SETITEM and a DoubleStorage tensor.
	var pkl bytes.Buffer
	pkl.Write([]byte{0x80, 0x02})
	pkl.WriteString("ccollections\nOrderedDict\n)Rq\x00")
	// key "w"
	pkl.Write([]byte{0x8c, 1, 'w'})
	// _rebuild_tensor_v2(storage(DoubleStorage, "0", "cpu", 4), 0, (2, 2), (2, 1))
	pkl.WriteString("ctorch._utils\n_rebuild_tensor_v2\n(")
	pkl.Write([]byte{'('})
	pkl.Write([]byte{0x8c, 7})
	pkl.WriteString("storage")
	pkl.WriteString("ctorch\nDoubleStorage\n")
	pkl.Write([]byte{0x8c, 1, '0'})
	pkl.Write([]byte{0x8c, 3})
	pkl.WriteString("cpu")
	pkl.Write([]byte{'K', 4, 't', 'Q'})
	pkl.Write([]byte{'K', 0})
	pkl.Write([]byte{'K', 2, 'K', 2, 0x86})
	pkl.Write([]byte{'K', 2, 'K', 1, 0x86})
	pkl.Write([]byte{'t', 'R', 's'})
	// Drop dict and retrieve via memo 0 to ensure SETITEM updated the memoized OrderedDict.
	pkl.Write([]byte{'h', 0, '.'})

	name := filepath.Join(t.TempDir(), "double.pth")
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	z := zip.NewWriter(f)
	w, err := z.Create("data.pkl")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(pkl.Bytes()); err != nil {
		t.Fatal(err)
	}
	raw, err := z.Create("data/0")
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1.5, -2.25, 3.75, 0.5}
	for _, v := range want {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(float64(v)))
		if _, err := raw.Write(buf[:]); err != nil {
			t.Fatal(err)
		}
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	got, err := Read(name)
	if err != nil {
		t.Fatal(err)
	}
	tensor, ok := got["w"]
	if !ok {
		t.Fatalf("missing key w in %v", got)
	}
	if !reflect.DeepEqual(tensor.Shape, []int{2, 2}) || !reflect.DeepEqual(tensor.Data, want) {
		t.Fatalf("got %+v want shape [2 2] data %v", tensor, want)
	}
}
