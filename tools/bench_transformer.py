"""Four pre-norm encoder layers, one full training step after warmup."""
import argparse
import time
import torch

p = argparse.ArgumentParser()
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--threads", type=int, default=12)
p.add_argument("--warmup", type=int, default=1)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(731)
device = torch.device(a.device)
model = torch.nn.Sequential(*[
    torch.nn.TransformerEncoderLayer(256, 4, 1024, dropout=0, activation="gelu", batch_first=True, norm_first=True)
    for _ in range(4)
]).to(device)
x = torch.randn(16, 256, 256, device=device)
opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)

def step():
    opt.zero_grad(set_to_none=True)
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
