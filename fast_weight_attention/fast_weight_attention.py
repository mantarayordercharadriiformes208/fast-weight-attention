from __future__ import annotations

import torch
from torch import nn, randn
from torch.nn import Module, Linear, ParameterDict, Sequential

from einops import einsum, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from adam_atan2_pytorch.muon_adam_atan2 import newtonschulz5
from adam_atan2_pytorch.polar_adam_atan2 import polar_express

# constants

def AttentionMemory(*, wq, wk, wv, wo):
    return dict(wq = wq, wk = wk, wv = wv, wo = wo)

def add_memories(mem1, mem2):
    return {k: mem1[k] + mem2[k] for k in mem1.keys()}

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class FastWeightAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_value_head = None,
        heads = 8,
        causal = True,
        max_learning_rate = 1e-2,
        muon_update = True,
        use_polar_express = False
    ):
        super().__init__()

        dim_value_head = default(dim_value_head, dim_head)

        self.norm = nn.RMSNorm(dim)

        # scale

        self.scale = dim_head ** -0.5

        self.causal = causal

        # memory parameters

        self.attn_memory = ParameterDict(dict(
            wq = randn(heads, dim, dim_head),
            wk = randn(heads, dim, dim_head),
            wv = randn(heads, dim, dim_value_head),
            wo = randn(heads, dim_value_head, dim),
        ))

        self.memory_keys = self.attn_memory.keys()

        # to optimizer related

        self.to_learning_rate = Sequential(
            Linear(dim, 1, bias = False),
            nn.Sigmoid()
        )

        self.max_learning_rate = max_learning_rate

        self.muon_update = muon_update
        self.use_polar_express = use_polar_express

        # target values
        # using the z-score as well as the gating as done for fast-weight PKM proposed by Sakana AI

        self.to_target_values = Sequential(
            Linear(dim, dim, bias = False),
            nn.LayerNorm(dim, elementwise_affine = False)
        )

        self.to_gates = Sequential(
            Linear(dim, heads, bias = False),
            Rearrange('... n h -> ... h n 1'),
            nn.Sigmoid()
        )

    def init_memories(self, batch):
        return {name: repeat(weights, '... -> b ...', b = batch) for name, weights in self.attn_memory.items()}

    def forward(
        self,
        tokens,
        return_next_memories = False,
        past_mem: AttentionMemory | None = None,
        detach_next_memories = False
    ):
        batch, scale = tokens.shape[0], self.scale

        # prenorm

        tokens = self.norm(tokens)

        # add the fast weight memories

        memory = self.init_memories(batch)

        if exists(past_mem):
            memory = add_memories(memory, past_mem)

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

        learning_rate = self.to_learning_rate(tokens) * self.max_learning_rate

        error = (target_values - pred_values_for_fast_weight) * learning_rate # flipped sign so no need to -grad at end

        # now go through the backwards pass of attention, using predicted loss to next target value (Sakana AI discovery)

        dout = einsum(error, wo, 'b n d, b h dh d -> b h n dh')

        delta = reduce(dout * out, '... d -> ... 1', 'sum')

        dv = einsum(attn, dout, 'b h i j, b h i dh -> b h j dh')

        dattn = einsum(v, dout, 'b h j dh, b h i dh -> b h i j')

        dscore = scale * attn * (dattn - delta)

        dq = einsum(k, dscore, 'b h j dh, b h i j -> b h i dh')
        dk = einsum(q, dscore, 'b h i dh, b h i j -> b h j dh')

        dwq = einsum(dq, tokens, 'b h i dh, b i d -> b h d dh')
        dwk = einsum(dk, tokens, 'b h j dh , b j d -> b h d dh')
        dwv = einsum(dv, tokens, 'b h j dh , b j d -> b h d dh')
        dwo = einsum(error, out, 'b n d, b h n dh -> b h dh d')

        if self.muon_update:
            update_fn = polar_express if self.use_polar_express else newtonschulz5
            dwv = update_fn(dwv)
            dwo = update_fn(dwo)

        next_fast_weights = AttentionMemory(wq = dwq, wk = dwk, wv = dwv, wo = dwo)

        if exists(past_mem):
            next_fast_weights = add_memories(next_fast_weights, past_mem)

        if detach_next_memories:
            next_fast_weights = {k: v.detach() for k, v in next_fast_weights.items()}

        return pred_values, next_fast_weights
