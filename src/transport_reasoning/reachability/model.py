"""Local incoming attention with explicit outer recurrence and inner transport."""
import math
import torch
from torch import nn
from .config import Config


def fixed_parameter(kind):
    target=32/15; k=torch.arange(1,5,dtype=torch.float64)
    def weights(logp):return (k*logp-torch.lgamma(k+1) if kind=='exponential' else (k-1)*logp).softmax(0)
    low,high=-40.,40.
    for _ in range(100):
        mid=(low+high)/2
        if (weights(mid)*k).sum()<target:low=mid
        else:high=mid
    return math.exp((low+high)/2)


def coefficients(kind,hops=4,parameter=None):
    if hops<1:raise ValueError('Positive hop count required')
    k=torch.arange(1,hops+1,dtype=torch.float64)
    if kind=='onehop' or hops==1:return torch.ones(1,dtype=torch.float64)
    if kind in ['binomial','learned']:
        logw=torch.tensor([math.lgamma(hops+1)-math.lgamma(i+1)-math.lgamma(hops-i+1) for i in range(1,hops+1)],dtype=torch.float64)
    elif kind in ['exponential','geometric']:
        p=fixed_parameter(kind) if parameter is None else parameter
        logw=k*math.log(p)-torch.lgamma(k+1) if kind=='exponential' else (k-1)*math.log(p)
    else:raise ValueError('Unknown filter')
    return logw.softmax(0)


class Block(nn.Module):
    def __init__(self,c,index):
        super().__init__();self.c=c;self.filter=c.filter;self.hops=c.hops;self.beta=c.inner_beta
        with torch.random.fork_rng():
            torch.manual_seed(c.seed+1009*(index+1))
            self.ln1=nn.LayerNorm(c.width);self.ln2=nn.LayerNorm(c.width)
            self.qkv=nn.Linear(c.width,3*c.width,bias=False);self.out=nn.Linear(c.width,c.width,bias=False)
            self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
        self.parameter=fixed_parameter(c.filter) if c.filter in ['exponential','geometric'] else None
        values=coefficients(c.filter,c.hops,self.parameter).float()
        if c.filter=='learned':self.theta=nn.Parameter(values.log())
        else:self.register_buffer('fixed_coeff',values)
        if c.inner_beta:
            with torch.random.fork_rng():
                torch.manual_seed(c.seed+20003+index)
                self.inner_ln=nn.LayerNorm(c.width)
                self.inner=nn.Sequential(nn.Linear(c.width,c.width),nn.GELU(),nn.Linear(c.width,c.width))
    def coeff(self):return self.theta.softmax(0) if hasattr(self,'theta') else self.fixed_coeff
    def forward(self,x,mask,valid):
        b,n,d=x.shape;h=self.c.heads
        q,k,v=self.qkv(self.ln1(x)).reshape(b,n,3,h,d//h).permute(2,0,3,1,4).unbind(0)
        a=((q@k.transpose(-1,-2))/math.sqrt(d//h)).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        z=v*valid[:,None,:,None];y=torch.zeros_like(v)
        for weight in self.coeff():
            z=a@z
            if self.beta:
                joined=z.transpose(1,2).reshape(b,n,d)
                joined=(joined+self.beta*self.inner(self.inner_ln(joined)))*valid[...,None]
                z=joined.reshape(b,n,h,d//h).transpose(1,2)
            y=y+weight*z
        x=x+self.out(y.transpose(1,2).reshape(b,n,d));x=x+self.ff(self.ln2(x))
        return x*valid[...,None]


class Encoder(nn.Module):
    def __init__(self,c):
        super().__init__();c.validate();self.c=c;self.outer_steps=c.outer_steps;self.observe_states=False;self.state_norms=[]
        with torch.random.fork_rng():
            torch.manual_seed(c.seed)
            self.embedding=nn.Linear(5,c.width);self.readout=nn.Linear(c.width,1);self.norm=nn.LayerNorm(c.width)
        self.blocks=nn.ModuleList([Block(c,i) for i in range(1 if c.share_weights else c.outer_steps)])
    def forward(self,model_inputs):
        if set(model_inputs)!={'x','attention_mask','node_valid','target_index'}:raise ValueError('Only the four model-input fields are accepted')
        valid=model_inputs['node_valid'];x=self.embedding(model_inputs['x'])*valid[...,None];self.state_norms=[]
        for i in range(self.outer_steps):
            x=self.blocks[0 if self.c.share_weights else i](x,model_inputs['attention_mask'],valid)
            if self.observe_states:
                self.state_norms.append(float(x.detach().float().norm(dim=-1).max().cpu()))
        target=x[torch.arange(len(x),device=x.device),model_inputs['target_index']]
        return self.readout(self.norm(target)).squeeze(-1)
    def extra_compute(self,hops=None,outer_steps=None):
        if outer_steps is not None:
            if not self.c.share_weights or self.c.model!='R32-one':raise ValueError('Only recurrent weights support extra outer iterations')
            self.outer_steps=outer_steps
        if hops is not None:
            if self.c.filter in ['onehop','learned'] and hops!=self.c.hops:raise ValueError('Extended filter unsupported')
            for block in self.blocks:
                if self.c.filter!='learned':block.fixed_coeff=coefficients(self.c.filter,hops,block.parameter).to(block.fixed_coeff)
                block.hops=hops
        self.observe_states=True
    def operations(self):
        hops=self.blocks[0].coeff().numel()
        return {'qkv_calls':self.outer_steps,'attention_calls':self.outer_steps,'transport_matmuls':self.outer_steps*hops,
                'inner_ffn_calls':self.outer_steps*hops if self.c.inner_beta else 0,'outer_ffn_calls':self.outer_steps,
                'unique_blocks':len(self.blocks),'parameters':sum(p.numel() for p in self.parameters())}
