"""Positive-hop filters; attention remains differentiable at every use."""
import math
import torch


def moments(c):
    c = torch.as_tensor(c, dtype=torch.float64)
    s = torch.arange(1, c.numel()+1, dtype=c.dtype, device=c.device)
    mu = (s*c).sum().item()
    nu = (s*(s-1)*c).sum().item()
    return {"coefficients": c.tolist(), "mu": mu, "nu": nu,
            "variance": nu+mu-mu*mu, "tail": c.flip(0).cumsum(0).flip(0).tolist()}


def coefficients(method, hops, target_mean=None):
    if hops < 1: raise ValueError("hops must be positive")
    if method in ("onehop", "recurrent"): return torch.ones(1, dtype=torch.float64)
    if hops == 1: return torch.ones(1, dtype=torch.float64)
    s = torch.arange(1, hops+1, dtype=torch.float64)
    log_bin = torch.tensor([math.lgamma(hops+1)-math.lgamma(i+1)-math.lgamma(hops-i+1)
                           for i in range(1, hops+1)], dtype=torch.float64)
    if method in ("binomial", "learned_polynomial"): return log_bin.softmax(0)
    target = target_mean if target_mean is not None else hops/(2*(1-2.**-hops))
    if not 1 < target < hops: raise ValueError("Target mean is outside the attainable interior")
    if method not in ("exponential_truncated", "finite_geometric"):
        raise ValueError("Unknown finite filter")
    def weights(log_param):
        logw = s*log_param-torch.lgamma(s+1) if method == "exponential_truncated" else (s-1)*log_param
        return logw.softmax(0)
    lo, hi = -40., 40.
    for _ in range(150):
        mid = (lo+hi)/2
        if (s*weights(mid)).sum() < target: lo = mid
        else: hi = mid
    c = weights((lo+hi)/2)
    if abs((s*c).sum().item()-target) > 1e-11: raise ArithmeticError("Mean matching failed")
    return c


def action(a, v, c):
    z, y = v, torch.zeros_like(v)
    for cs in c.unbind(-1):
        z = a @ z
        y = y + cs * z
    return y


def binomial_stable(a, v, hops):
    z, u = v, torch.zeros_like(v)
    for _ in range(hops):
        t = a @ z
        z, u = (z+t)/2, (u+t)/2
    return u/(1-2.**-hops)


def exponential_full_action(a, v, lam, tolerance=1e-12, max_hops=10000):
    """Positive action series with an explicit omitted mass certificate."""
    if not 0 < lam <= 100 or not 0 < tolerance < 1:
        raise ValueError("Require 0 < lambda <= 100 and 0 < tolerance < 1")
    c = lam/math.expm1(lam)
    z = a @ v
    y = c*z
    mass, degree = c, 1
    while 1-mass > tolerance and degree < max_hops:
        degree += 1
        c *= lam/degree
        z = a @ z
        y = y+c*z
        mass += c
    tail = max(0., 1-mass)
    if tail > tolerance: raise ArithmeticError("Exponential series tolerance not attained")
    return y, {"degree": degree, "omitted_mass": tail,
               "row_infinity_operator_bound": tail, "renormalized": False}


def sparse_action(a, v, c):
    if a.layout == torch.strided: raise ValueError("Expected a sparse matrix")
    z, y = v, torch.zeros_like(v)
    for cs in c:
        z = torch.sparse.mm(a, z)
        y = y+cs*z
    return y
