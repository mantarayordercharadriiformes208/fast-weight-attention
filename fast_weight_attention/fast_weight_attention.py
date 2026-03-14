from __future__ import annotations
from functools import partial

import torch
from torch import nn, randn
from torch.nn import Module, Linear, ParameterDict, Sequential

from einx import multiply
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

from adam_atan2_pytorch.muon_adam_atan2 import newtonschulz5
from adam_atan2_pytorch.polar_adam_atan2 import polar_express

# constants

LinearNoBias = partial(Linear, bias = False)

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
        max_muon_learning_rate = 1e-1,
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
            LinearNoBias(dim, 2 if muon_update else 1),
            nn.Sigmoid()
        )

        self.max_learning_rate = max_learning_rate

        # muon related

        self.muon_update = muon_update
        self.use_polar_express = use_polar_express

        self.max_muon_learning_rate = max_muon_learning_rate

        # target values
        # using the z-score as well as the gating as done for fast-weight PKM proposed by Sakana AI

        self.to_target_values = Sequential(
            LinearNoBias(dim, dim),
            nn.LayerNorm(dim, elementwise_affine = False)
        )

        self.to_gates = Sequential(
            LinearNoBias(dim, heads),
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
        batch, scale, muon_update = tokens.shape[0], self.scale, self.muon_update

        # prenorm

        tokens = self.norm(tokens)
        gates = self.to_gates(tokens)

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

        out = out * gates

        pred_values = einsum(out, wo, 'b h n dh, b h dh d -> b n d')

        if not return_next_memories:
            return pred_values

        target_values = self.to_target_values(tokens[..., 1:, :])

        # some slicing so subsequent backwards pass is readable

        tokens = tokens[..., :-1, :]

        pred_values_for_fast_weight = pred_values[..., :-1, :]

        out = out[..., :-1, :]

        gates = gates[..., :-1, :]

        attn = attn[..., :-1, :-1]

        q, k, v = q[..., :-1, :], k[..., :-1, :], v[..., :-1, :]

        # per token learning rate related

        learning_rate = self.to_learning_rate(tokens) * self.max_learning_rate

        if muon_update:
            learning_rate, muon_learning_rate = learning_rate.unbind(dim = -1)
        else:
            learning_rate = rearrange(learning_rate, '... 1 -> ...')

        # mse error
        # flipped sign so no need to -grad at end

        error = (target_values - pred_values_for_fast_weight)

        # now go through the backwards pass of attention, using predicted loss to next target value (Sakana AI discovery)

        dout = einsum(error, wo, 'b n d, b h dh d -> b h n dh')

        du = dout * gates

        delta = reduce(dout * out, '... d -> ... 1', 'sum')

        dv = einsum(attn, du, 'b h i j, b h i dh -> b h j dh')

        dattn = einsum(v, du, 'b h j dh, b h i dh -> b h i j')

        dscore = scale * attn * (dattn - delta)

        dq = einsum(k, dscore, 'b h j dh, b h i j -> b h i dh')
        dk = einsum(q, dscore, 'b h i dh, b h i j -> b h j dh')

        # apply learning rates

        if muon_update:
            tokens_for_dwqk = multiply('b n d, b n', tokens, learning_rate)
            tokens_for_dwv = multiply('b n d, b n', tokens, muon_learning_rate)
            out_for_dwo = multiply('b h n d, b n', out, muon_learning_rate)
        else:
            tokens_for_dwqk = tokens
            tokens_for_dwv = tokens
            out_for_dwo = out

        # get the next memories

        dwq = einsum(dq, tokens_for_dwqk, 'b h i dh, b i d -> b h d dh')
        dwk = einsum(dk, tokens_for_dwqk, 'b h j dh , b j d -> b h d dh')
        dwv = einsum(dv, tokens_for_dwv, 'b h j dh , b j d -> b h d dh')
        dwo = einsum(error, out_for_dwo, 'b n d, b h n dh -> b h dh d')

        if muon_update:
            update_fn = polar_express if self.use_polar_express else newtonschulz5
            dwv = update_fn(dwv)
            dwo = update_fn(dwo)

        # prep next memories

        next_mems = AttentionMemory(wq = dwq, wk = dwk, wv = dwv, wo = dwo)

        if exists(past_mem):
            next_mems = add_memories(next_mems, past_mem)

        if detach_next_memories:
            next_mems = {k: v.detach() for k, v in next_mems.items()}

        return pred_values, next_mems
