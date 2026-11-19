//go:build windows && amd64

package directml

import (
	"fmt"

	"github.com/yh2237/gograd/gpu"
	"unsafe"
)

// A copy list handles arbitrary element offsets and rank-six strides without
// reading values on the host. Scatter writes exactly the requested elements;
// unlike strided operator outputs, padding cannot overwrite unrelated aliases.
func (d *nativeDriver) copyView(out, src *allocation, v gpu.View, scatter bool) error {
	n, _ := gpu.Elements(v.Shape...)
	run := 1
	for i := len(v.Shape) - 1; i >= 0; i-- {
		if v.Shape[i] > 1 && v.Strides[i] != run {
			break
		}
		run *= v.Shape[i]
	}
	regions := (n + run - 1) / run
	if regions > maxCopyRegions {
		return fmt.Errorf("%w: DirectML scatter of %d elements needs %d copy regions", gpu.ErrUnsupported, n, regions)
	}
	vc := gpu.View{
		Buffer:  v.Buffer,
		Shape:   append([]int(nil), v.Shape...),
		Strides: append([]int(nil), v.Strides...),
		Offset:  v.Offset,
	}
	return d.enqueue(queuedStep{kind: stepCopyView, out: out, src: src, view: vc, run: run, regions: regions, scatter: scatter})
}

func (d *nativeDriver) recordCopyView(out, src *allocation, v gpu.View, run int, scatter bool) {
	n, _ := gpu.Elements(v.Shape...)
	for flat := 0; flat < n; flat += run {
		index, rest := v.Offset, flat
		for axis := len(v.Shape) - 1; axis >= 0; axis-- {
			index += rest % v.Shape[axis] * v.Strides[axis]
			rest /= v.Shape[axis]
		}
		dest, source := uint64(flat)*4, uint64(index)*4
		if scatter {
			dest, source = source, dest
		}
		call(d.command, 15, uintptr(unsafe.Pointer(out.native.(*object))), uintptr(dest), uintptr(unsafe.Pointer(src.native.(*object))), uintptr(source), uintptr(run)*4)
	}
}
