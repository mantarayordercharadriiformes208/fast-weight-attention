from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, randn, randint, tensor, is_tensor
from torch.nn import Module, Linear, ParameterDict, Sequential

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
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

def transpose(t):
    return t.transpose(-1, -2)

# helper modules

class Scale(Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, t):
        return t * self.scale

# Muon related - Keller Jordan

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):

    if t.ndim < 2:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = transpose(t)

    t, packed_shape = pack([t], '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ transpose(t)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t, = unpack(t, packed_shape, '* i j')

    if should_transpose:
        t = transpose(t)

    return t

# classes

class FastWeightAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        max_learning_rate = 1e-2,
        muon_update = True
    ):
        super().__init__()

        self.norm = nn.RMSNorm(dim)

        # scale

        self.scale = dim_head ** -0.5

        self.causal = causal

        # memory parameters

        self.attn_memory = ParameterDict(dict(
            wq = randn(heads, dim, dim_head),
            wk = randn(heads, dim, dim_head),
            wv = randn(heads, dim, dim_head),
            wo = randn(heads, dim_head, dim),
        ))

        self.memory_keys = self.attn_memory.keys()

        # to optimizer related

        self.to_learning_rate = Sequential(
            nn.Linear(dim, 1, bias = False),
            nn.Sigmoid(),
            Scale(max_learning_rate)
        )

        self.muon_update = muon_update

        # target values
        # using the z-score as well as the gating as done for fast-weight PKM proposed by Sakana AI

        self.to_target_values = Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.LayerNorm(dim, elementwise_affine = False)
        )

        self.to_gates = Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('... n h -> ... h n 1'),
            nn.Sigmoid()
        )

    def init_memories(self, batch):
        return {name: repeat(weights, '... -> b ...', b = batch) for name, weights in self.attn_memory.items()}

    def forward(
        self,
        tokens,
        return_next_memories = False,
        past_mem: AttentionMemory | None = None
    ):
        batch, scale = tokens.shape[0], self.scale

        # prenorm

        tokens = self.norm(tokens)

        # add the fast weight memories

        memory = self.init_memories(batch)

        if exists(past_mem):
            memory = {name: (memory[name] + past_mem[name]) for name in self.memory_keys}

        # get the memories

        wq, wk, wv, wo = tuple(memory[name] for name in ('wq', 'wk', 'wv', 'wo'))

        # attention

        q = einsum(tokens, wq, 'b n d, b h d dh -> b h n dh')
        k = einsum(tokens, wk, 'b n d, b h d dh -> b h n dh')
        v = einsum(tokens, wv, 'b n d, b h d dh -> b h n dh')

        score = einsum(q, k, 'b h i dh, b h j dh -> b h i j') * scale

        if self.causal:
            i, j = score.shape[-2:]
            causal_mask = torch.ones((i, j), device = score.device, dtype = torch.bool).triu(j - i + 1)
            score = score.masked_fill(causal_mask, -torch.finfo(score.dtype).max)

        attn = score.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j dh -> b h i dh')

        pred_values = einsum(out, wo, 'b h n dh, b h dh d -> b n d')

        if not return_next_memories:
            return pred_values

        target_values = self.to_target_values(tokens[..., 1:, :])

        # some slicing so subsequent backwards pass is readable

        tokens = tokens[..., :-1, :]

        pred_values_for_fast_weight = pred_values[..., :-1, :]

        out = out[..., :-1, :]

        attn = attn[..., :-1, :-1]

        q, k, v = q[..., :-1, :], k[..., :-1, :], v[..., :-1, :]

        # mse error

        learning_rate = self.to_learning_rate(tokens)

        error = (target_values - pred_values_for_fast_weight) * learning_rate # flipped sign so no need to -grad at end

        # now go through the backwards pass of attention, using predicted loss to next target value (Sakana AI discovery)

        dout = einsum(error, wo, 'b n d, b h dh d -> b h n dh')

        dwo = einsum(error, out, 'b n d, b h n dh -> h dh d')

        delta = reduce(dout * out, '... d -> ... 1', 'sum')

        dv = einsum(attn, dout, 'b h i j, b h i dh -> b h j dh')

        dattn = einsum(v, dout, 'b h j dh, b h i dh -> b h i j')

        dscore = scale * attn * (dattn - delta)

        dq = einsum(k, dscore, 'b h j dh, b h i j -> b h i dh')
        dk = einsum(q, dscore, 'b h i dh, b h i j -> b h j dh')

        dwq = einsum(dq, tokens, 'b h i dh, b i d -> h d dh')
        dwk = einsum(dk, tokens, 'b h j dh , b j d -> h d dh')
        dwv = einsum(dv, tokens, 'b h j dh , b j d -> h d dh')

        if self.muon_update:
            dwv = newtonschulz5(dwv)
            dwo = newtonschulz5(dwo)

        # returning

        next_fast_weights = AttentionMemory(wq = dwq, wk = dwk, wv = dwv, wo = dwo)

        return pred_values, next_fast_weights
