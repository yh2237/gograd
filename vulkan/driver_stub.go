//go:build (!windows && !linux) || android || !amd64

package vulkan

import (
	"fmt"
	"runtime"

	"github.com/yh2237/gograd/gpu"
)

func openDriver() (driver, error) {
	return nil, fmt.Errorf("%w: Vulkan loader is not implemented for %s/%s", gpu.ErrUnavailable, runtime.GOOS, runtime.GOARCH)
}
