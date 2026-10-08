"""Small PyTorch CPU fixture for reusable transformer graph compositions."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
rotary = torch.tensor([math.sin(i*.071) for i in range(1*2*4*8)],dtype=torch.float32).reshape(1,2,4,8)
position = torch.arange(4,dtype=torch.float32)
inv = 1.0/(160000.0**(torch.arange(0,8,2,dtype=torch.float32)/8))
freq = torch.outer(position,inv)
cos,sin=freq.cos()[None,None],freq.sin()[None,None]
a,b=rotary.chunk(2,dim=-1)
rotated=torch.cat([a*cos-b*sin,b*cos+a*sin],dim=-1)
ge=torch.tensor([math.cos(i*.13) for i in range(24)],dtype=torch.float32).reshape(2,12)
g0,g1=ge.chunk(2,dim=-1)
sx=torch.tensor([math.sin(i*.2) for i in range(8)],dtype=torch.float32).reshape(2,4)
w1=torch.tensor([math.cos(i*.07)*.3 for i in range(24)],dtype=torch.float32).reshape(6,4)
w2=torch.tensor([math.sin(i*.05)*.2 for i in range(24)],dtype=torch.float32).reshape(4,6)
w3=torch.tensor([math.cos(i*.09)*.25 for i in range(24)],dtype=torch.float32).reshape(6,4)
k=torch.tensor([math.cos(i*.031) for i in range(1*2*4*8)],dtype=torch.float32).reshape(1,2,4,8)
v=torch.tensor([math.sin(i*.043) for i in range(1*2*4*8)],dtype=torch.float32).reshape(1,2,4,8)
valid=[True,True,False,True]
mask=torch.tensor([[[[valid[j] and abs(i-j)<=1 for j in range(4)] for i in range(4)]]])
embedding=torch.tensor([math.cos(i*.12) for i in range(9*4)],dtype=torch.float32).reshape(9,4)
ids=[1,4,0,8]
payload={
    "rotary_input":rotary.flatten().tolist(),"rotary_output":rotated.flatten().tolist(),
    "geglu_input":ge.flatten().tolist(),"geglu_output":(F.gelu(g0)*g1).flatten().tolist(),
    "swiglu_input":sx.flatten().tolist(),"w1":w1.flatten().tolist(),"w2":w2.flatten().tolist(),"w3":w3.flatten().tolist(),
    "swiglu_output":F.linear(F.silu(F.linear(sx,w1))*F.linear(sx,w3),w2).flatten().tolist(),
    "attention_q":rotary.flatten().tolist(),"attention_k":k.flatten().tolist(),"attention_v":v.flatten().tolist(),
    "attention_output":F.scaled_dot_product_attention(rotary,k,v,attn_mask=mask).flatten().tolist(),
    "embedding_weight":embedding.flatten().tolist(),"embedding_ids":ids,"embedding_output":F.embedding(torch.tensor(ids),embedding).flatten().tolist(),
}
Path("testdata/speech_transformer.json").write_text(json.dumps(payload,separators=(",",":"))+"\n",encoding="utf-8")
