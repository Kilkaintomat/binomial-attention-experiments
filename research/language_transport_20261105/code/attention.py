"""Positive-hop transport, adapted from the audited project filters.

No edge oracle, inner FFN, gate, detached routing, or KV cache. Qwen's outer
residual/FFN, QK normalization, RoPE and grouped-query expansion are retained.
"""
import copy
import math
import types

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


def coefficients(method, r):
    if r < 1:
        raise ValueError('r must be positive')
    if method == 'baseline':
        return torch.ones(1, dtype=torch.float64)
    if method == 'power':
        c = torch.zeros(r, dtype=torch.float64); c[-1] = 1
        return c
    s = torch.arange(1, r + 1, dtype=torch.float64)
    if method == 'binomial':
        return torch.tensor([math.lgamma(r+1)-math.lgamma(k+1)-math.lgamma(r-k+1)
                             for k in range(1, r+1)], dtype=torch.float64).softmax(0)
    if method != 'geometric':
        raise ValueError(method)
    if r == 1:
        return torch.ones(1, dtype=torch.float64)
    target = r / (2 * (1 - 2.**-r))
    lo, hi = -40., 40.
    for _ in range(100):
        mid = (lo + hi) / 2
        if float((s * ((s-1)*mid).softmax(0)).sum()) < target:
            lo = mid
        else:
            hi = mid
    return ((s-1)*((lo+hi)/2)).softmax(0)


def action(a, v, method, r):
    if method == 'baseline' or r == 1:
        return a @ v
    if method == 'binomial':
        # Existing binomial_stable recurrence: no powers of an n x n matrix,
        # no subtraction of nearly equal states and no exponentially large gain.
        z, u = v, torch.zeros_like(v)
        for _ in range(r):
            t = a @ z
            z, u = (z + t) * .5, (u + t) * .5
        return u / (1 - 2.**-r)
    if method == 'power':
        for _ in range(r):
            v = a @ v
        return v
    c = coefficients(method, r).to(device=v.device, dtype=v.dtype)
    z, out = v, torch.zeros_like(v)
    for w in c:
        z = a @ z
        out = out + w*z
    return out


def forward(self, hidden_states, position_embeddings, attention_mask,
            past_key_value=None, cache_position=None, **kwargs):
    if past_key_value is not None:
        raise ValueError('This campaign deliberately uses full-prefix, cache-free inference')
    shape = hidden_states.shape[:-1]
    headshape = (*shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(headshape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(headshape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(headshape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)
    scores = (q @ k.transpose(-1, -2)) * self.scaling
    n = q.shape[-2]
    allowed = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.dtype == torch.bool:
            raise ValueError('Expected the additive Qwen causal/padding mask')
        scores = scores + attention_mask[..., :n]
    a = scores.softmax(-1, dtype=torch.float32).nan_to_num(0).to(q.dtype)
    if self.attention_dropout:
        raise ValueError('Attention dropout must be zero')
    out = action(a, v, self.lt_method, self.lt_r)
    if self.lt_capture:
        self.lt_attention = a.detach().float().cpu()
    out = out.transpose(1, 2).contiguous().reshape(*shape, -1)
    return self.o_proj(out), (a if kwargs.get('output_attentions') else None)


def root(model):
    return model.get_base_model() if hasattr(model, 'peft_config') else model


def install(model, method, selected, r=8, loops=2):
    base = root(model)
    if method == 'looped':
        # Repeat the entire four-block backbone; QKV and FFN are recomputed.
        # Shallow clones retain exactly shared parameters and independent indices.
        base.lt_base_layers=len(base.model.layers)
        base.lt_loops=1
        set_loops(model,loops)
        return
    # The baseline also uses this implementation in selected layers, so r=1
    # identity and timing differences do not depend on switching kernels there.
    for i in selected:
        attn = base.model.layers[i].self_attn
        attn.lt_method, attn.lt_r, attn.lt_capture = method, r, False
        attn.forward = types.MethodType(forward, attn)


def set_loops(model,loops):
    base=root(model)
    if loops<1:raise ValueError('Positive loop count required')
    if base.lt_loops==loops:return
    original=list(base.model.layers[:base.lt_base_layers]);layers=[]
    for _ in range(loops):
        for layer in original:
            clone=copy.copy(layer);clone._modules=dict(layer._modules)
            attn=copy.copy(layer.self_attn);attn._modules=dict(attn._modules)
            attn.layer_idx=len(layers);clone.self_attn=attn;layers.append(clone)
    base.model.layers=nn.ModuleList(layers)
    base.config.num_hidden_layers=len(layers);base.lt_loops=loops


def change_r(model, r, method=None):
    for m in model.modules():
        if hasattr(m, 'lt_method'):
            m.lt_r = r
            if method is not None:
                m.lt_method = method


def operator_row(a, query, method, r):
    """One row only, all heads; not a matrix power. Evaluation diagnostic."""
    row = torch.zeros((*a.shape[:-2], 1, a.shape[-1]), dtype=a.dtype, device=a.device)
    row[..., 0, query] = 1
    result = torch.zeros_like(row)
    for c in coefficients(method, r).to(a):
        row = row @ a
        result += c * row
    return result.squeeze(-2)
