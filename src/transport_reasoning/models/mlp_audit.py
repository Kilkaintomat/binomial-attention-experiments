"""Small exact identities and explicitly labeled frozen-gate approximations."""
import torch


def direct(x,k,v,bk,bv,eta,steps):
    for _ in range(steps): x=x+eta*(torch.relu(x@k.T+bk)@v+bv)
    return x


def slot_recurrence(x,k,v,bk,bv,eta,steps):
    z=x@k.T+bk; r=v@k.T; total=torch.zeros_like(x)
    for _ in range(steps):
        gate=torch.relu(z)
        total=total+gate@v
        z=z+eta*(gate@r+bv@k.T)
    return x+eta*total+steps*eta*bv


def frozen_affine(x,k,v,bk,bv):
    gate=(x@k.T+bk>0).to(x.dtype)
    b=k.T@(gate[:,None]*v)
    c=(bk*gate)@v+bv
    augmented=torch.zeros((x.numel()+1,x.numel()+1),dtype=x.dtype,device=x.device)
    augmented[:-1,:-1]=b; augmented[-1,:-1]=c
    return b,c,augmented


def audit(seed=1901):
    torch.manual_seed(seed)
    dtype=torch.float64
    x=torch.randn(4,dtype=dtype)*.1; k=torch.randn(7,4,dtype=dtype)*.1
    v=torch.randn(7,4,dtype=dtype)*.1; bk=torch.randn(7,dtype=dtype)*.1; bv=torch.randn(4,dtype=dtype)*.01
    args=[a.clone().requires_grad_() for a in (x,k,v,bk,bv)]
    y=direct(*args,.1,8); z=slot_recurrence(*args,.1,8)
    gy=torch.autograd.grad(y.square().sum(),args,retain_graph=True)
    gz=torch.autograd.grad(z.square().sum(),args)
    b,c,bhat=frozen_affine(x,k,v,bk,bv)
    frozen=x.clone()
    for _ in range(8): frozen=frozen+.1*(frozen@b+c)
    xhat=torch.cat([x,torch.ones(1,dtype=dtype)])
    polynomial=xhat@torch.linalg.matrix_power(torch.eye(5,dtype=dtype)+.1*bhat,8)
    target=xhat@torch.matrix_exp(.8*bhat)
    euler=[float((xhat@torch.linalg.matrix_power(torch.eye(5,dtype=dtype)+(.8/r)*bhat,r)-target).norm()) for r in [8,16,32,64]]
    state=x.clone(); initial=(state@k.T+bk)>0; flips=[]
    for _ in range(8):
        state=direct(state,k,v,bk,bv,.1,1); flips.append(float((((state@k.T+bk)>0)!=initial).double().mean()))
    return {"direct_slot_max_error":float((y-z).abs().max()),
            "direct_slot_max_gradient_error":max(float((a-b).abs().max()) for a,b in zip(gy,gz)),
            "frozen_polynomial_max_error":float((frozen-polynomial[:-1]).abs().max()),
            "nonlinear_to_frozen_error":float((y-frozen).norm()),"frozen_evaluation_error":float((frozen-polynomial[:-1]).norm()),
            "euler_to_exp_errors_fixed_tau":euler,"relu_gate_flip_fraction":flips,
            "scope":"small float64 algebra check, not task evidence or GPU speed evidence"}
