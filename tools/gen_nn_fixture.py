#!/usr/bin/env python3
"""Generate the float32 general-nn parity fixture. Run: python tools/gen_nn_fixture.py"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(17)
torch.set_num_threads(1)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 3)
        self.conv = nn.Conv1d(3, 3, 3, dilation=2, padding=2)
        self.output = nn.Linear(3, 2)
    def forward(self, x):
        h = self.input(x)
        h = h + F.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2), approximate="tanh")
        return self.output(h)

net = Net()
x = torch.randn(2, 5, 2)
target = torch.randn(2, 5, 2)
mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
channels = torch.tensor([1.0, 0.7])
prediction = net(x)
w = mask[..., None] * channels
loss = ((prediction - target).square() * w).sum() / w.expand_as(prediction).sum()
loss.backward()
before = {name: value.detach().flatten().tolist() for name, value in net.named_parameters()}
grads = {name: value.grad.detach().flatten().tolist() for name, value in net.named_parameters()}
torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01)
optimizer.step()
after = {name: value.detach().flatten().tolist() for name, value in net.named_parameters()}
fixture = dict(input=x.flatten().tolist(), target=target.flatten().tolist(), mask=mask.flatten().tolist(),
               channels=channels.tolist(), forward=prediction.detach().flatten().tolist(), loss=loss.item(),
               parameters=before, gradients=grads, updated=after)
Path("testdata/nn_conv_step.json").write_text(json.dumps(fixture, indent=2) + "\n")
