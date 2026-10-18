package gpu

import (
	"errors"
	"reflect"
	"testing"
)

func TestViewValidation(t *testing.T) {
	b := &testBackend{name: "views"}
	buffer, err := NewBuffer(b, 24, nil, func() {})
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Close()
	for _, v := range []View{
		{Buffer: buffer, Shape: []int{4, 2, 3}, Strides: []int{1, 12, 4}},
		{Buffer: buffer, Shape: []int{2, 2}, Strides: []int{12, 4}, Offset: 5},
		{Buffer: buffer, Shape: []int{2, 3, 4}, Strides: []int{0, 0, 0}},
		{Buffer: buffer, Shape: []int{0, 3}, Strides: []int{3, 1}, Offset: 24},
		{Buffer: buffer, Offset: 23},
	} {
		if _, err := v.Validate(b); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range []View{
		{Buffer: buffer, Shape: []int{2, 3}, Strides: []int{24, 1}},
		{Buffer: buffer, Shape: []int{2}, Strides: []int{-1}},
		{Buffer: buffer, Shape: []int{1}, Strides: nil},
		{Buffer: buffer, Offset: -1}, {Buffer: buffer, Offset: 24},
		{Buffer: buffer, Shape: []int{0}, Strides: []int{1}, Offset: 25},
	} {
		if _, err := v.Validate(b); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("%+v: %v", v, err)
		}
	}
	good := View{Buffer: buffer, Shape: []int{4, 2, 3}, Strides: []int{1, 12, 4}}
	if _, err := good.ValidateWrite(b); err != nil {
		t.Fatal(err)
	}
	for _, v := range []View{{Buffer: buffer, Shape: []int{2, 3}, Strides: []int{0, 1}}, {Buffer: buffer, Shape: []int{2, 2}, Strides: []int{1, 1}}} {
		if _, err := v.ValidateWrite(b); !errors.Is(err, ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	if _, err := good.Validate(&testBackend{name: "views"}); !errors.Is(err, ErrBackendMismatch) {
		t.Fatal(err)
	}
	buffer.Close()
	if _, err := good.Validate(b); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestViewShapes(t *testing.T) {
	for _, tc := range []struct{ a, b, want []int }{
		{[]int{2, 1, 3, 5}, []int{1, 4, 5, 7}, []int{2, 4, 3, 7}},
		{[]int{3, 5}, []int{5, 7}, []int{3, 7}},
		{[]int{0, 3, 5}, []int{1, 5, 7}, []int{0, 3, 7}},
	} {
		got, err := MatMulShape(tc.a, tc.b)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%v %v: %v %v", tc.a, tc.b, got, err)
		}
	}
	if _, err := MatMulShape([]int{2, 3}, []int{4, 5}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := MatMulShape([]int{2, -1}, []int{-1, 3}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	v := View{Shape: []int{1, 3}, Strides: []int{3, 1}, Offset: 2}
	out, err := v.BroadcastTo([]int{2, 4, 3})
	if err != nil || !reflect.DeepEqual(out.Strides, []int{0, 0, 1}) || out.Offset != 2 {
		t.Fatalf("%+v %v", out, err)
	}
	if _, err := v.BroadcastTo([]int{2, 4}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
}
