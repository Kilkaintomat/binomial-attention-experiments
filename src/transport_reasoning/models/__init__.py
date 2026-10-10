"""Common pre-norm encoder, with explicit fixed-routing and refresh boundaries."""
import math
import torch
from torch import nn
from transport_reasoning.filters import coefficients, action, moments


class Block(nn.Module):
    def __init__(self,c,index):
        super().__init__()
        self.c=c
        # Named RNG streams keep common parameter tensors equal across methods.
        with torch.random.fork_rng():
            torch.manual_seed(c.seed+1009*(index+1))
            self.ln1=nn.LayerNorm(c.width); self.ln2=nn.LayerNorm(c.width)
            self.qkv=nn.Linear(c.width,3*c.width,bias=False)
            self.out=nn.Linear(c.width,c.width,bias=False)
            self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
        coeff=coefficients(c.method,c.hops).float()
        if c.method=="learned_polynomial": self.theta=nn.Parameter(coeff.log())
        else: self.register_buffer("fixed_coeff",coeff)
        if c.inner_beta:
            with torch.random.fork_rng():
                torch.manual_seed(c.seed+20003+index)
                self.inner_ln=nn.LayerNorm(c.width)
                self.inner=nn.Sequential(nn.Linear(c.width,c.inner_width or c.width),nn.GELU(),
                                         nn.Linear(c.inner_width or c.width,c.width))

    def coeff(self):
        return self.theta.softmax(-1) if hasattr(self,"theta") else self.fixed_coeff

    def forward(self,x,mask):
        b,n,d=x.shape; h=self.c.heads; hd=d//h
        q,k,v=self.qkv(self.ln1(x)).reshape(b,n,3,h,hd).permute(2,0,3,1,4).unbind(0)
        logits=(q @ k.transpose(-1,-2))/math.sqrt(hd)
        a=logits.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        cs=self.coeff()
        if not self.c.inner_beta:
            y=action(a,v,cs)
        else:
            z=v; y=torch.zeros_like(v)
            for weight in cs:
                hz=(a@z).transpose(1,2).reshape(b,n,d)
                zz=hz+self.c.inner_beta*self.inner(self.inner_ln(hz))
                z=zz.reshape(b,n,h,hd).transpose(1,2)
                y=y+weight*z
        x=x+self.out(y.transpose(1,2).reshape(b,n,d))
        return x+self.ff(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        inputs=8 if c.task=="maze" else 4*c.size+2
        outputs=1 if c.task=="maze" else c.size
        with torch.random.fork_rng():
            torch.manual_seed(c.seed)
            self.embedding=nn.Linear(inputs,c.width)
            self.readout=nn.Linear(c.width,outputs)
            self.norm=nn.LayerNorm(c.width)
        self.blocks=nn.ModuleList([Block(c,i) for i in range(1 if c.method=="recurrent" else c.layers)])

    def forward(self,batch):
        x=self.embedding(batch["x"])
        if self.c.task=="maze":
            # Absolute sinusoidal coordinates extend to larger grids without new embeddings.
            coords=batch["x"][:,:,4:6]
            freq=torch.exp(-math.log(10000)*torch.arange(self.c.width//4,device=x.device)/(self.c.width//4))
            angles=coords[:,:,:,None]*freq
            position=torch.cat([angles.sin(),angles.cos()],dim=-1).reshape_as(x)
            x=x+position
        if self.c.method=="recurrent":
            for _ in range(self.c.recurrent_steps): x=self.blocks[0](x,batch["mask"])
        else:
            for block in self.blocks: x=block(x,batch["mask"])
        return self.readout(self.norm(x))

    def filter_summary(self):
        return [moments(b.coeff().detach().cpu()) for b in self.blocks]

    def parameter_count(self): return sum(p.numel() for p in self.parameters())
