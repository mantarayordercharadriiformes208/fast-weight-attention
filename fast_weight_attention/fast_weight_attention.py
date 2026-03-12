from __future__ import annotations
from collections import namedtuple

import torch
from torch import randn, randint, tensor, is_tensor
from torch.nn import Module, Linear, ParameterDict

import einx
from einops import einsum, pack, unpack
from einops.layers.torch import Rearrange

# constants

AttentionMemory = namedtuple('AttentionMemory', (
    'wq', # (heads, dim, dim_head)
    'wk', # (heads, dim, dim_head)
    'wv', # (heads, dim, dim_head)
    'wo'  # (heads, dim_head, dim)
))

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

        # scale

        self.scale = dim_head ** -0.5

        # memory parameters

        self.attn_memory = ParameterDict(dict(
            wq = randn(heads, dim, dim_head),
            wk = randn(heads, dim, dim_head),
            wv = randn(heads, dim, dim_head),
            wo = randn(heads, dim_head, dim),
        ))

        self.memory_keys = self.attn_memory.keys()

    def forward(
        self,
        tokens,
        return_next_memories = False,
        past_mem: AttentionMemory | None = None
    ):
        # add the fast weight memories 

        memory = self.attn_memory

        if exists(past_mem):
            memory = {(memory[name] + past_mem[name]) for name in self.memory_keys}

        # get the memories

        wq, wk, wv, wo = tuple(memory[name] for name in ('wq', 'wk', 'wv', 'wo'))

        # attention

        q = einsum(tokens, wq, 'b n d, h d dh -> b h n dh')
        k = einsum(tokens, wk, 'b n d, h d dh -> b h n dh')
        v = einsum(tokens, wv, 'b n d, h d dh -> b h n dh')

        q = q * self.scale

        sim = einsum(q, k, 'b h i dh, b h j dh -> b h i j')

        attn = sim.softmax(dim = -1)

        aggregated = einsum(attn, v, 'b h i j, b h j dh -> b h i dh')

        retrieved = einsum(aggregated, wo, 'b h n dh, h dh d -> b n d')

        return retrieved, None
