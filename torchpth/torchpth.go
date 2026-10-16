package torchpth

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"path"
	"strings"
)

type Tensor struct {
	Shape []int
	Data  []float32
}
type storage struct {
	kind, key string
	count     int
}
type global string
type mark struct{}
type pair struct {
	key   string
	value any
}
type ordered []pair

type decoder struct {
	data   []byte
	pos    int
	stack  []any
	memo   map[int]any
	files  map[string]*zip.File
	prefix string
}

// Read loads a torch.save zip checkpoint. Nested state dictionaries are joined with dots;
// a standalone tensor is returned under the empty name.
func Read(filename string) (map[string]Tensor, error) {
	z, e := zip.OpenReader(filename)
	if e != nil {
		return nil, e
	}
	defer z.Close()
	files := map[string]*zip.File{}
	var pickle *zip.File
	for _, f := range z.File {
		files[f.Name] = f
		if strings.HasSuffix(f.Name, "/data.pkl") {
			pickle = f
		}
	}
	if pickle == nil {
		return nil, fmt.Errorf("torchpth: missing data.pkl")
	}
	r, e := pickle.Open()
	if e != nil {
		return nil, e
	}
	data, e := io.ReadAll(r)
	r.Close()
	if e != nil {
		return nil, e
	}
	d := decoder{data: data, memo: map[int]any{}, files: files, prefix: strings.TrimSuffix(pickle.Name, "data.pkl")}
	root, e := d.run()
	if e != nil {
		return nil, e
	}
	out := map[string]Tensor{}
	var visit func(any, string) error
	visit = func(v any, name string) error {
		switch x := v.(type) {
		case Tensor:
			out[name] = x
		case map[string]any:
			for k, v := range x {
				n := k
				if name != "" {
					n = name + "." + k
				}
				if e := visit(v, n); e != nil {
					return e
				}
			}
		case ordered:
			for _, p := range x {
				n := p.key
				if name != "" {
					n = name + "." + n
				}
				if e := visit(p.value, n); e != nil {
					return e
				}
			}
		default:
			return fmt.Errorf("torchpth: unsupported value at %s", name)
		}
		return nil
	}
	if e = visit(root, ""); e != nil {
		return nil, e
	}
	return out, nil
}

func (d *decoder) bytes(n int) ([]byte, error) {
	if n < 0 || d.pos+n > len(d.data) {
		return nil, fmt.Errorf("torchpth: truncated pickle")
	}
	b := d.data[d.pos : d.pos+n]
	d.pos += n
	return b, nil
}
func (d *decoder) pop() (any, error) {
	if len(d.stack) == 0 {
		return nil, fmt.Errorf("torchpth: stack underflow")
	}
	v := d.stack[len(d.stack)-1]
	d.stack = d.stack[:len(d.stack)-1]
	return v, nil
}
func (d *decoder) items() ([]any, error) {
	i := len(d.stack) - 1
	for i >= 0 {
		if _, ok := d.stack[i].(mark); ok {
			v := append([]any(nil), d.stack[i+1:]...)
			d.stack = d.stack[:i]
			return v, nil
		}
		i--
	}
	return nil, fmt.Errorf("torchpth: missing mark")
}
func asInt(v any) (int, error) {
	n, ok := v.(int)
	if !ok {
		return 0, fmt.Errorf("torchpth: expected int")
	}
	return n, nil
}
func asInts(v any) ([]int, error) {
	a, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("torchpth: expected tuple")
	}
	out := make([]int, len(a))
	for i, x := range a {
		var e error
		out[i], e = asInt(x)
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}
func (d *decoder) run() (any, error) {
	for d.pos < len(d.data) {
		b, _ := d.bytes(1)
		op := b[0]
		switch op {
		case 0x80:
			_, e := d.bytes(1)
			if e != nil {
				return nil, e
			}
		case '.':
			return d.pop()
		case '(':
			d.stack = append(d.stack, mark{})
		case ')':
			d.stack = append(d.stack, []any{})
		case '}':
			d.stack = append(d.stack, map[string]any{})
		case 0x88:
			d.stack = append(d.stack, true)
		case 0x89:
			d.stack = append(d.stack, false)
		case 'N':
			d.stack = append(d.stack, nil)
		case 'K':
			v, e := d.bytes(1)
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, int(v[0]))
		case 'M':
			v, e := d.bytes(2)
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, int(binary.LittleEndian.Uint16(v)))
		case 'J':
			v, e := d.bytes(4)
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, int(int32(binary.LittleEndian.Uint32(v))))
		case 'X':
			v, e := d.bytes(4)
			if e != nil {
				return nil, e
			}
			n := binary.LittleEndian.Uint32(v)
			if n > 1<<25 {
				return nil, fmt.Errorf("torchpth: oversized string")
			}
			s, e := d.bytes(int(n))
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, string(s))
		case 0x8c:
			v, e := d.bytes(1)
			if e != nil {
				return nil, e
			}
			s, e := d.bytes(int(v[0]))
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, string(s))
		case 'c':
			start := d.pos
			for d.pos < len(d.data) && d.data[d.pos] != '\n' {
				d.pos++
			}
			if d.pos == len(d.data) {
				return nil, fmt.Errorf("torchpth: bad global")
			}
			module := string(d.data[start:d.pos])
			d.pos++
			start = d.pos
			for d.pos < len(d.data) && d.data[d.pos] != '\n' {
				d.pos++
			}
			if d.pos == len(d.data) {
				return nil, fmt.Errorf("torchpth: bad global")
			}
			name := string(d.data[start:d.pos])
			d.pos++
			g := module + " " + name
			switch g {
			case "collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch._utils _rebuild_parameter", "torch FloatStorage", "torch HalfStorage", "torch BFloat16Storage":
				d.stack = append(d.stack, global(g))
			default:
				return nil, fmt.Errorf("torchpth: forbidden global %s", g)
			}
		case 'q':
			v, e := d.bytes(1)
			if e != nil {
				return nil, e
			}
			if len(d.stack) == 0 {
				return nil, fmt.Errorf("torchpth: empty memo")
			}
			d.memo[int(v[0])] = d.stack[len(d.stack)-1]
		case 'r':
			v, e := d.bytes(4)
			if e != nil {
				return nil, e
			}
			d.memo[int(binary.LittleEndian.Uint32(v))] = d.stack[len(d.stack)-1]
		case 'h':
			v, e := d.bytes(1)
			if e != nil {
				return nil, e
			}
			x, ok := d.memo[int(v[0])]
			if !ok {
				return nil, fmt.Errorf("torchpth: missing memo")
			}
			d.stack = append(d.stack, x)
		case 'j':
			v, e := d.bytes(4)
			if e != nil {
				return nil, e
			}
			x, ok := d.memo[int(binary.LittleEndian.Uint32(v))]
			if !ok {
				return nil, fmt.Errorf("torchpth: missing memo")
			}
			d.stack = append(d.stack, x)
		case 't':
			v, e := d.items()
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, v)
		case 0x85, 0x86, 0x87:
			n := int(op - 0x84)
			if len(d.stack) < n {
				return nil, fmt.Errorf("torchpth: tuple underflow")
			}
			v := append([]any(nil), d.stack[len(d.stack)-n:]...)
			d.stack = d.stack[:len(d.stack)-n]
			d.stack = append(d.stack, v)
		case 'Q':
			v, e := d.pop()
			if e != nil {
				return nil, e
			}
			a, ok := v.([]any)
			if !ok || len(a) < 5 || a[0] != "storage" {
				return nil, fmt.Errorf("torchpth: invalid storage")
			}
			g, ok := a[1].(global)
			if !ok {
				return nil, fmt.Errorf("torchpth: invalid storage type")
			}
			key, ok := a[2].(string)
			if !ok {
				return nil, fmt.Errorf("torchpth: invalid storage key")
			}
			count, e := asInt(a[4])
			if e != nil {
				return nil, e
			}
			d.stack = append(d.stack, storage{string(g), key, count})
		case 'R':
			args, e := d.pop()
			if e != nil {
				return nil, e
			}
			fn, e := d.pop()
			if e != nil {
				return nil, e
			}
			g, ok := fn.(global)
			if !ok {
				return nil, fmt.Errorf("torchpth: invalid reduce")
			}
			a, ok := args.([]any)
			if !ok {
				return nil, fmt.Errorf("torchpth: invalid arguments")
			}
			var result any
			switch g {
			case "collections OrderedDict":
				result = ordered{}
			case "torch._utils _rebuild_tensor_v2":
				if len(a) < 4 {
					return nil, fmt.Errorf("torchpth: short tensor args")
				}
				s, ok := a[0].(storage)
				if !ok {
					return nil, fmt.Errorf("torchpth: invalid tensor storage")
				}
				off, e := asInt(a[1])
				if e != nil {
					return nil, e
				}
				shape, e := asInts(a[2])
				if e != nil {
					return nil, e
				}
				stride, e := asInts(a[3])
				if e != nil {
					return nil, e
				}
				result, e = d.tensor(s, off, shape, stride)
				if e != nil {
					return nil, e
				}
			case "torch._utils _rebuild_parameter":
				if len(a) == 0 {
					return nil, fmt.Errorf("torchpth: short parameter")
				}
				result = a[0]
			default:
				return nil, fmt.Errorf("torchpth: unsupported reduce %s", g)
			}
			d.stack = append(d.stack, result)
		case 's':
			v, e := d.pop()
			if e != nil {
				return nil, e
			}
			k, e := d.pop()
			if e != nil {
				return nil, e
			}
			obj, e := d.pop()
			if e != nil {
				return nil, e
			}
			key, ok := k.(string)
			if !ok {
				return nil, fmt.Errorf("torchpth: non-string key")
			}
			switch m := obj.(type) {
			case map[string]any:
				m[key] = v
			case ordered:
				m = append(m, pair{key, v})
				obj = m
			default:
				return nil, fmt.Errorf("torchpth: not a dict")
			}
			d.stack = append(d.stack, obj)
		case 'u':
			items, e := d.items()
			if e != nil {
				return nil, e
			}
			obj, e := d.pop()
			if e != nil {
				return nil, e
			}
			if len(items)%2 != 0 {
				return nil, fmt.Errorf("torchpth: odd dict items")
			}
			for i := 0; i < len(items); i += 2 {
				k, ok := items[i].(string)
				if !ok {
					return nil, fmt.Errorf("torchpth: non-string key")
				}
				switch m := obj.(type) {
				case map[string]any:
					m[k] = items[i+1]
				case ordered:
					m = append(m, pair{k, items[i+1]})
					obj = m
				default:
					return nil, fmt.Errorf("torchpth: not a dict")
				}
			}
			d.stack = append(d.stack, obj)
		case 'b':
			_, e := d.pop()
			if e != nil {
				return nil, e
			}
		default:
			return nil, fmt.Errorf("torchpth: unsupported opcode %#x", op)
		}
	}
	return nil, fmt.Errorf("torchpth: missing STOP")
}

func (d *decoder) tensor(s storage, off int, shape, stride []int) (Tensor, error) {
	if len(shape) != len(stride) || off < 0 || s.count < 0 {
		return Tensor{}, fmt.Errorf("torchpth: invalid tensor shape")
	}
	n := 1
	for _, dim := range shape {
		if dim < 0 || dim > 1<<25 || n > 1<<28/max(1, dim) {
			return Tensor{}, fmt.Errorf("torchpth: oversized tensor")
		}
		n *= dim
	}
	f := d.files[path.Join(d.prefix, "data", s.key)]
	if f == nil {
		f = d.files[strings.ReplaceAll(d.prefix, "/", "\\")+"data/"+s.key]
	}
	if f == nil {
		return Tensor{}, fmt.Errorf("torchpth: missing storage %s", s.key)
	}
	r, e := f.Open()
	if e != nil {
		return Tensor{}, e
	}
	raw, e := io.ReadAll(r)
	r.Close()
	if e != nil {
		return Tensor{}, e
	}
	size := 4
	if s.kind != "torch FloatStorage" {
		size = 2
	}
	if len(raw) != s.count*size {
		return Tensor{}, fmt.Errorf("torchpth: storage length mismatch")
	}
	result := Tensor{Shape: shape, Data: make([]float32, n)}
	for i := 0; i < n; i++ {
		index := off
		remaining := i
		for j := len(shape) - 1; j >= 0; j-- {
			if shape[j] > 0 {
				index += (remaining % shape[j]) * stride[j]
				remaining /= shape[j]
			}
		}
		if index < 0 || index >= s.count {
			return Tensor{}, fmt.Errorf("torchpth: tensor view outside storage")
		}
		switch s.kind {
		case "torch FloatStorage":
			result.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
		case "torch BFloat16Storage":
			result.Data[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[index*2:])) << 16)
		case "torch HalfStorage":
			result.Data[i] = half(binary.LittleEndian.Uint16(raw[index*2:]))
		}
	}
	return result, nil
}
func half(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	ex := uint32(h>>10) & 31
	mant := uint32(h & 1023)
	if ex == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		v := float32(mant) * float32(math.Ldexp(1, -24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	if ex == 31 {
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (ex+112)<<23 | mant<<13)
}
