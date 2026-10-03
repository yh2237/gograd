"""Generate a small deterministic PyTorch Acoustic training-step fixture."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(733)
torch.set_num_threads(1)


class Acoustic(nn.Module):
    def __init__(self, speakers=3, hidden=4, outputs=5):
        super().__init__()
        self.phone = nn.Embedding(46, 64)
        self.speaker = nn.Embedding(speakers, 64)
        self.inp = nn.Conv1d(260, hidden, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(hidden, hidden, 5, dilation=d, padding=2*d) for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16))
        self.norms = nn.ModuleList(nn.GroupNorm(1, hidden) for _ in self.blocks)
        self.film = nn.ModuleList(nn.Linear(64, hidden*2) for _ in self.blocks)
        self.out = nn.Conv1d(hidden, outputs, 1)

    def forward(self, ids, cont, speaker):
        s = self.speaker(speaker)
        b, t, _ = ids.shape
        x = torch.cat((self.phone(ids).flatten(2), s[:, None, :].expand(-1, t, -1), cont), 2).transpose(1, 2)
        h = self.inp(x)
        for block, norm, film in zip(self.blocks, self.norms, self.film):
            scale, shift = film(s).chunk(2, 1)
            h = h + F.gelu(norm(block(h)) * (1 + scale[:, :, None]) + shift[:, :, None])
        return self.out(h).transpose(1, 2)


def fixture():
    model = Acoustic()
    ids = torch.randint(0, 46, (2, 3, 3))
    cont = torch.randn(2, 3, 4)
    speaker = torch.tensor([0, 2])
    target = torch.randn(2, 3, 5)
    target[1, 2] = math.nan
    params = {name: {"shape": list(p.shape), "data": p.detach().flatten().tolist()} for name, p in model.named_parameters()}
    pred = model(ids, cont, speaker)
    loss = (pred - torch.nan_to_num(target)).abs()[~torch.isnan(target[..., 0])].mean()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    opt.zero_grad()
    loss.backward()
    grads = {name: p.grad.detach().flatten().tolist() for name, p in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    after = {name: p.detach().flatten().tolist() for name, p in model.named_parameters()}
    return {"ids": ids.flatten().tolist(), "speaker": speaker.tolist(), "cont": cont.flatten().tolist(), "target": target.nan_to_num().flatten().tolist(), "mask": [not math.isnan(v) for v in target[..., 0].flatten().tolist()], "params": params, "pred": pred.detach().flatten().tolist(), "loss": loss.item(), "grads": grads, "norm": norm.item(), "after": after}


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "testdata" / "acoustic_step.json"
    path.write_text(json.dumps(fixture(), allow_nan=False))
    print(path)
