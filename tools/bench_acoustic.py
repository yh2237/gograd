"""One full Acoustic training step; outputs timing after a warmup step."""
import argparse
import time
import torch
from gen_acoustic_fixture import Acoustic

p = argparse.ArgumentParser()
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--threads", type=int, default=12)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(733)
device = torch.device(a.device)
model = Acoustic(speakers=102, hidden=384, outputs=88).to(device)
ids = torch.randint(0, 46, (24, 400, 3), device=device)
cont = torch.randn(24, 400, 4, device=device)
speakers = torch.randint(0, 102, (24,), device=device)
target = torch.randn(24, 400, 88, device=device)
target[:, -10:] = torch.nan
opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)

def step():
    opt.zero_grad(set_to_none=True)
    pred = model(ids, cont, speakers)
    loss = (pred - torch.nan_to_num(target)).abs()[~torch.isnan(target[..., 0])].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return loss.item()

step()
start = time.perf_counter()
loss = step()
print(f"torch {a.device}: {(time.perf_counter()-start)*1000:.3f} ms loss {loss:.6f}", flush=True)
