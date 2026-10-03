"""Emit deterministic PyTorch reference values for the Go transformer tests."""
import json
import sys
import torch
import torch.nn.functional as F

device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
torch.manual_seed(731)
torch.set_num_threads(1)

def data(t):
    return t.detach().cpu().contiguous().view(-1).tolist()

def item(t):
    return {"shape": list(t.shape), "data": data(t), "grad": data(t.grad) if t.grad is not None else None}

out = {}
def keep(i, seed, p):
    x = (i ^ seed) & 0xffffffff
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xffffffff
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xffffffff
    x ^= x >> 16
    return x >= int(p * 4294967296)

x = torch.randn(2, 3, 4, device=device, requires_grad=True)
up = torch.randn_like(x)
p, seed = 0.25, 123
mask = torch.tensor([float(keep(i, seed, p)) for i in range(x.numel())], device=device).reshape_as(x)
y = x * mask / (1 - p)
(y * up).sum().backward()
out["dropout"] = {"x": item(x), "up": item(up), "y": item(y)}

for name, function in (("softmax", torch.softmax), ("log_softmax", torch.log_softmax)):
    x = torch.randn(2, 3, 4, device=device, requires_grad=True)
    up = torch.randn_like(x)
    y = function(x, dim=1)
    (y * up).sum().backward()
    out[name] = {"x": item(x), "up": item(up), "y": item(y)}

x = torch.randn(4, 5, device=device, requires_grad=True)
target = torch.tensor([1, -100, 3, 0], device=device)
y = F.cross_entropy(x, target, ignore_index=-100)
y.backward()
out["cross_entropy"] = {"x": item(x), "target": data(target), "y": item(y)}

x = torch.randn(2, 3, 4, device=device, requires_grad=True)
w = torch.randn(4, device=device, requires_grad=True)
b = torch.randn(4, device=device, requires_grad=True)
up = torch.randn_like(x)
y = F.layer_norm(x, (4,), w, b, 1e-5)
(y * up).sum().backward()
out["layer_norm"] = {"x": item(x), "w": item(w), "b": item(b), "up": item(up), "y": item(y)}

a = torch.randn(2, 1, 3, 4, device=device, requires_grad=True)
b = torch.randn(1, 3, 4, 5, device=device, requires_grad=True)
up = torch.randn(2, 3, 3, 5, device=device)
y = a @ b
(y * up).sum().backward()
out["batched_matmul"] = {"a": item(a), "b": item(b), "up": item(up), "y": item(y)}

q = torch.randn(2, 3, 4, 5, device=device, requires_grad=True)
k = torch.randn(2, 3, 6, 5, device=device, requires_grad=True)
v = torch.randn(2, 3, 6, 7, device=device, requires_grad=True)
mask = torch.randn(1, 1, 4, 6, device=device, requires_grad=True) * 0.1
mask.retain_grad()
up = torch.randn(2, 3, 4, 7, device=device)
y = torch.softmax(q @ k.transpose(-2, -1) / (5 ** 0.5) + mask, -1) @ v
(y * up).sum().backward()
out["attention"] = {"q": item(q), "k": item(k), "v": item(v), "mask": item(mask), "up": item(up), "y": item(y)}

w = torch.randn(6, 4, device=device, requires_grad=True)
ids = torch.tensor([[0, 2, 2], [5, 0, 3]], device=device)
up = torch.randn(2, 3, 4, device=device)
y = F.embedding(ids, w, padding_idx=0)
(y * up).sum().backward()
out["embedding_padding"] = {"w": item(w), "ids": data(ids), "up": item(up), "y": item(y)}

module = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0, activation="gelu", batch_first=True, norm_first=True, device=device) for _ in range(2)])
x = torch.randn(2, 4, 8, device=device, requires_grad=True)
up = torch.randn_like(x)
params = {f"{i}.{name}": item(param) for i, layer in enumerate(module) for name, param in layer.named_parameters()}
y = x
for layer in module:
    y = layer(y)
loss = (y * up).sum()
optimizer = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.01)
loss.backward()
grads = {f"{i}.{name}": data(param.grad) for i, layer in enumerate(module) for name, param in layer.named_parameters()}
optimizer.step()
step = {f"{i}.{name}": data(param) for i, layer in enumerate(module) for name, param in layer.named_parameters()}
out["transformer"] = {"x": item(x), "up": item(up), "y": item(y), "params": params, "grads": grads, "step": step}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f)
