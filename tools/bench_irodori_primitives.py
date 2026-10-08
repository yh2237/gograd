"""CPU RMSNorm comparison for `go test ./irodori -bench BenchmarkRMSNorm`."""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from irodori_tts.model import RMSNorm

torch.set_num_threads(1)
x = torch.zeros((256, 1024), dtype=torch.float32)
layer = RMSNorm(1024, eps=1e-6).eval()
with torch.inference_mode():
    for _ in range(20):
        layer(x)
    samples = []
    for _ in range(100):
        start = time.perf_counter_ns()
        layer(x)
        samples.append(time.perf_counter_ns() - start)
print(f"PyTorch CPU RMSNorm (256x1024), 1 thread: median={statistics.median(samples):.0f} ns/op")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    x_cuda = x.to(device)
    layer_cuda = layer.to(device)
    with torch.inference_mode():
        for _ in range(20):
            layer_cuda(x_cuda).cpu()
        samples = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            layer_cuda(x_cuda).cpu()
            torch.cuda.synchronize()
            samples.append(time.perf_counter_ns() - start)
    print(f"PyTorch CUDA RMSNorm (256x1024), synchronized host readback: median={statistics.median(samples):.0f} ns/op")
