package autograd

import (
	"os"
	"os/exec"
	"sync"
	"testing"
)

var (
	pytorchOnce      sync.Once
	pytorchAvailable bool
)

// requirePyTorchは、PyTorchの参照値を作るPythonが無い環境（CIなど）でテストを飛ばす。
// GOGRAD_REQUIRE_PYTORCH=1なら飛ばさずに失敗させる。
func requirePyTorch(t *testing.T) {
	t.Helper()
	pytorchOnce.Do(func() {
		pytorchAvailable = exec.Command("python", "-c", "import numpy, torch").Run() == nil
	})
	if !pytorchAvailable && os.Getenv("GOGRAD_REQUIRE_PYTORCH") != "1" {
		t.Skip("python with numpy and torch is unavailable")
	}
}
