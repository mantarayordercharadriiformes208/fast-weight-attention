import pytest

def test_mem():
    import torch

    from fast_weight_attention import FastWeightAttention

    mem = FastWeightAttention(512)

    tokens = torch.randn(1, 64, 512)

    past_mem = None

    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)
    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)
    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)

    assert retrieved.shape == tokens.shape
