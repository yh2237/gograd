"""Four pre-norm encoder layers, one full training step after warmup."""
import argparse
import time
import torch

p = argparse.ArgumentParser()
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--threads", type=int, default=12)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--graph", action="store_true")
p.add_argument("--bf16", action="store_true")
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(731)
device = torch.device(a.device)
model = torch.nn.Sequential(*[
    torch.nn.TransformerEncoderLayer(256, 4, 1024, dropout=0, activation="gelu", batch_first=True, norm_first=True)
    for _ in range(4)
]).to(device)
x = torch.randn(16, 256, 256, device=device)
opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, capturable=a.graph)

def step():
    opt.zero_grad(set_to_none=not a.graph)
    with torch.autocast(device_type=a.device, dtype=torch.bfloat16, enabled=a.bf16):
        y = model(x)
        loss = (y * y).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

for _ in range(a.warmup):
    step()
start = time.perf_counter()
step()
print(f"torch {a.device} transformer: {(time.perf_counter()-start)*1000:.3f} ms", flush=True)
if a.graph:
    if device.type != "cuda":
        raise ValueError("graph capture requires CUDA")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.zero_grad(set_to_none=False)
        with torch.autocast(device_type=a.device, dtype=torch.bfloat16, enabled=a.bf16):
            static_y = model(x)
            static_loss = (static_y * static_y).mean()
        static_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
    start = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    loss = static_loss.item()
    print(f"torch cuda transformer graph replay: {(time.perf_counter()-start)*1000:.3f} ms loss {loss:.6f}", flush=True)
