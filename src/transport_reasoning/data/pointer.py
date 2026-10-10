"""Independent reimplementation of the paper's pointer task, not original code."""
import math
import numpy as np
import torch
from torch import nn
from transport_reasoning.filters import coefficients,action


def views(seed):
    gen=torch.Generator().manual_seed(seed+6001)
    return [torch.linalg.qr(torch.randn(32,32,generator=gen)).Q for _ in range(2)]


def generate(batch,depth,width,generator,query_view,key_view):
    n=(depth+1)*width
    entities=torch.randn(batch,n,32,generator=generator)
    entities=entities/entities.norm(dim=-1,keepdim=True)*math.sqrt(32)
    target_index=torch.arange(n).expand(batch,n).clone()
    for level in range(1,depth+1):
        for b in range(batch): target_index[b,level*width:(level+1)*width]=(level-1)*width+torch.randperm(width,generator=generator)
    destination=entities.gather(1,target_index[:,:,None].expand(-1,-1,32))
    q=destination@query_view; k=entities@key_view
    labels=torch.zeros(batch,n,dtype=torch.long)
    for b in range(batch): labels[b,:width]=torch.arange(width)[torch.randperm(width,generator=generator)]%4
    for level in range(1,depth+1):
        labels[:,level*width:(level+1)*width]=labels.gather(1,target_index[:,level*width:(level+1)*width])
    v=torch.zeros(batch,n,4); v[:,:width]=torch.nn.functional.one_hot(labels[:,:width],4).float()
    mask=torch.tril(torch.ones(n,n,dtype=torch.bool),diagonal=-1)
    mask[:width]=False
    for i in range(width): mask[i,i]=True
    return q,k,v,mask,labels[:,-width:],target_index


class Pointer(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.q=nn.Linear(32,32,bias=False); self.k=nn.Linear(32,32,bias=False)
            nn.init.orthogonal_(self.q.weight); nn.init.orthogonal_(self.k.weight)

    def forward(self,data,method,hops,width):
        q,k,v,mask,labels,pointers=data
        a=(2*self.q(q)@self.k(k).transpose(-1,-2)/math.sqrt(32)).masked_fill(~mask,-torch.inf).softmax(-1)
        result=action(a,v,coefficients(method,hops).to(v))[:,-width:]
        return result,a


def metrics(model,data,method,hops,width):
    y,a=model(data,method,hops,width)
    labels=data[4]; mass=y.sum(-1); correct=y.gather(-1,labels[:,:,None]).squeeze(-1)
    conditional=correct/mass.clamp_min(1e-30)
    edges=data[5][:,width:]
    return {"conditional_accuracy":float((y.argmax(-1)==labels).float().mean()),
            "raw_correct_class_mass":float(correct.mean()),"terminal_mass":float(mass.mean()),
            "unreached_accuracy":float((torch.cat([y,(1-mass)[:,:,None]],-1).argmax(-1)==labels).float().mean()),
            "loss":float(-conditional.clamp_min(1e-30).log().mean()),
            "edge_accuracy":float((a[:,width:].argmax(-1)==edges).float().mean()),
            "edge_probability":float(a[:,width:].gather(-1,edges[:,:,None]).mean())}
