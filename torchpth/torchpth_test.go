package torchpth

import (
	"archive/zip"
	"encoding/json"
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
