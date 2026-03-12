# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "fast-weight-attention",
#     "torch",
#     "tqdm",
#     "x-mlps-pytorch"
# ]
# ///

import fire
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from einops import rearrange

from fast_weight_attention import FastWeightAttention
from x_mlps_pytorch import Feedforwards

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# model

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth = 1,
        dim_value_head = 32,
        causal = True
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                FastWeightAttention(
                    dim = dim,
                    dim_head = 32,
                    dim_value_head = dim_value_head,
                    heads = 4,
                    causal = causal
                ),
                Feedforwards(dim, depth = 1)
            ]) for _ in range(depth)
        ])

        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x, past_mems = None, return_next_memories = False):
        h = self.embed(x)

        past_mems = default(past_mems, [None] * len(self.layers))
        next_mems = []

        for (attn, ff), past_mem in zip(self.layers, past_mems):
            if return_next_memories:
                attn_out, next_mem = attn(h, past_mem = past_mem, return_next_memories = True)
                next_mems.append(next_mem)
            else:
                attn_out = attn(h, past_mem = past_mem)

            h = h + attn_out
            h = h + ff(h)

        if return_next_memories:
            return self.head(h), next_mems

        return self.head(h)

# chunked forward helpers

def chunked_forward(model, x, labels, chunk_size, use_memory = True):
    """Both baseline and memory use identical chunked processing.
    The only difference: memory passes past_mems, baseline does not."""

    past_mems = None
    total_loss = 0.
    preds_list = []

    for chunk_idx in range(0, x.shape[1], chunk_size):
        end_idx = min(chunk_idx + chunk_size, x.shape[1])
        x_chunk = x[:, chunk_idx:end_idx]
        labels_chunk = labels[:, chunk_idx:end_idx]

        preds, next_mems = model(x_chunk, past_mems = past_mems, return_next_memories = True)

        total_loss = total_loss + F.cross_entropy(
            rearrange(preds, 'b n d -> (b n) d'),
            rearrange(labels_chunk, 'b n -> (b n)')
        )

        preds_list.append(preds)

        # only carry memories forward for the memory condition
        past_mems = next_mems if use_memory else None

    num_chunks = (x.shape[1] + chunk_size - 1) // chunk_size
    return total_loss / num_chunks, torch.cat(preds_list, dim = 1)

# training

def train(
    seed = 42,
    num_tokens = 8,
    dim = 64,
    depth = 1,
    dim_value_head = 32,
    causal = True,
    half_len = 4,
    batch_size = 16,
    num_batches = 2500,
    lr = 3e-3,
    chunk_size = 4,
    eval_batches = 50,
    eval_every = 100
):
    assert chunk_size <= half_len, 'chunk size must be less than or equal to half sequence length'

    results = dict()

    for use_memory in (False, True):
        torch.manual_seed(seed)

        model = MemorizingModel(num_tokens, dim, depth = depth, dim_value_head = dim_value_head, causal = causal)
        optim = Adam(model.parameters(), lr = lr)

        label = 'Memory' if use_memory else 'Baseline'
        pbar = tqdm(range(num_batches), desc = label)
        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (batch_size, half_len))
            seq = torch.cat((half, half), dim = -1)

            x, labels = seq[:, :-1], seq[:, 1:]

            loss, _ = chunked_forward(model, x, labels, chunk_size, use_memory = use_memory)

            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()

            if i % eval_every == 0 or i >= (num_batches - eval_batches):
                model.eval()
                with torch.no_grad():
                    _, all_preds = chunked_forward(model, x, labels, chunk_size, use_memory = use_memory)
                    preds = all_preds.argmax(dim = -1)
                    acc = (preds[:, half_len:] == labels[:, half_len:]).float().mean()

                    if i >= (num_batches - eval_batches):
                        last_accs.append(acc.item())

                pbar.set_postfix(loss = f'{loss:.3f}', acc = f'{acc.item():.3f}')

        results[label] = sum(last_accs) / len(last_accs)

    # report

    print(f'\n{"-" * 40}')
    for label, acc in results.items():
        print(f'  {label}: {acc:.1%}')
    print(f'{"-" * 40}')

    memory_acc = results['Memory']
    baseline_acc = results['Baseline']
    advantage = memory_acc - baseline_acc
    print(f'\n  {"✅ Memory works!" if advantage > 0.20 else "❌ No clear advantage."}')

if __name__ == '__main__':
    fire.Fire(train)
