package irodori

import (
	"strings"

	"github.com/yh2237/gograd/autograd"
)

// PreloadDenoiser retains the 12 DiT blocks and timestep/output projections
// in host memory for repeated rectified-flow calls. The other checkpoint
// sections remain streamed, keeping resident memory around 1.5 GB.
func PreloadDenoiser(state *autograd.SafeTensorFile) error {
	var names []string
	for _, name := range state.Names() {
		if strings.HasPrefix(name, "blocks.") || strings.HasPrefix(name, "cond_module.") || strings.HasPrefix(name, "in_proj.") || strings.HasPrefix(name, "out_norm.") || strings.HasPrefix(name, "out_proj.") {
			names = append(names, name)
		}
	}
	return state.PreloadF32(names...)
}
