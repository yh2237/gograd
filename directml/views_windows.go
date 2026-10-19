//go:build windows && amd64

package directml

import (
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
	return d.submit(func() {
		d.transition(src, 0, stateCopySource)
		d.transition(out, 0, stateCopyDest)
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
		d.transition(src, stateCopySource, 0)
		d.transition(out, stateCopyDest, 0)
	})
}
