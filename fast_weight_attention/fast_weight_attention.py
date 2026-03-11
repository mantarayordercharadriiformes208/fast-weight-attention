from __future__ import annotations

import torch
from torch.nn import Module, Linear

from einops import einsum
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# Muon related - Keller Jordan

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):

    if t.ndim > 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, packed_shape = pack([t], '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t, = unpack(t, packed_shape, '* i j')

    if should_transpose:
        t = t.transpose(-1, -2)

    return t

# classes

class FastWeightAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.to_qk = Linear(dim, dim_inner * 2, bias = False)
        self.to_v = Linear(dim, dim_inner, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens
    ):

        q, k = self.to_qk(tokens).chunk(2, dim = -1)
        v = self.to_v(tokens)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

