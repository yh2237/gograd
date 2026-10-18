//go:build !windows || !amd64

package directml

import (
	"fmt"
	"github.com/yh2237/gograd/gpu"
)

func openDriver() (driver, error) {
	return nil, fmt.Errorf("%w: DirectML requires Windows/amd64", gpu.ErrUnavailable)
}
