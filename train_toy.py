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
from termcolor import colored

from fast_weight_attention import FastWeightAttention, ChunkManager
from x_mlps_pytorch import Feedforwards

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def print_header(char='-', length=40):
    print(char * length)

# model

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth = 1,
        dim_head = 32,
        dim_value_head = 32,
        heads = 4,
        causal = True,
        muon_update = True,
        use_polar_express = False,
        max_learning_rate = 1e-2,
        chunk_size = 4,
        use_forget_gate = True
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                ChunkManager(
                    FastWeightAttention(
                        dim = dim,
                        dim_head = dim_head,
                        dim_value_head = dim_value_head,
                        heads = heads,
                        causal = causal,
                        muon_update = muon_update,
                        use_polar_express = use_polar_express,
                        max_learning_rate = max_learning_rate
                    ),
                    chunk_size = chunk_size,
                    use_forget_gate = use_forget_gate
                ),
                Feedforwards(dim, depth = 1)
            ]) for _ in range(depth)
        ])

        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x, past_mems = None, return_next_memories = False, ablate_mem = False):
        h = self.embed(x)

        past_mems = default(past_mems, [None] * len(self.layers))
        next_mems = []

        for (attn, ff), past_mem in zip(self.layers, past_mems):
            if return_next_memories:
                attn_out, next_mem = attn(h, past_mem = past_mem, return_next_memories = True, ablate_mem = ablate_mem)
                next_mems.append(next_mem)
            else:
                attn_out = attn(h, past_mem = past_mem, ablate_mem = ablate_mem)

            h = h + attn_out
            h = h + ff(h)

        if return_next_memories:
            return self.head(h), next_mems

        return self.head(h)

# training

def train(
    seed = 42,
    num_tokens = 8,
    dim = 64,
    depth = 1,
    dim_head = 32,
    dim_value_head = 32,
    heads = 4,
    causal = True,
    batch_size = 16,
    num_batches = 2500,
    lr = 3e-3,
    chunk_size = 4,
    half_len = 8,
    eval_batches = 50,
    eval_every = 100,
    memory_only = False,
    muon_update = True,
    use_polar_express = True,
    max_learning_rate = 1e-3,
    use_forget_gate = False
):
    assert chunk_size <= half_len, 'chunk size must be less than or equal to half sequence length'

    total_len = half_len * 2

    print('')
    print(colored(f'Fast Weight Memory Toy Task', 'cyan', attrs=['bold']))
    print_header('-')
    print(f'The model must learn an auto-regressive sequence of length {total_len}')
    print(f'consisting of a random chunk of length {half_len} repeated twice.')
    print(f'Since it processes this in chunks of {chunk_size}, it must carry information')
    print(f'across chunks via its fast weight memories to predict the second half.')
    print_header('-')
    print(colored(f'Hyperparameters:', 'cyan'))
    print(f'  dim={dim}, heads={heads}, depth={depth}, forget_gate={use_forget_gate}')
    if muon_update:
        print(f'  Update Rule: Muon (polar_express={use_polar_express}) | max_lr={max_learning_rate}')
    else:
        print(f'  Update Rule: Plain | lr_base={lr} | max_fast_lr={max_learning_rate}')
    print_header('-')
    print('')

    results = dict()

    conditions = (True,) if memory_only else (True, False)

    for use_memory in conditions:
        torch.manual_seed(seed)

        model = MemorizingModel(
            num_tokens, dim, depth = depth,
            dim_head = dim_head, dim_value_head = dim_value_head,
            heads = heads, causal = causal,
            muon_update = muon_update,
            use_polar_express = use_polar_express,
            max_learning_rate = max_learning_rate,
            chunk_size = chunk_size,
            use_forget_gate = use_forget_gate
        )
        optim = Adam(model.parameters(), lr = lr)

        label = 'Memory' if use_memory else 'Baseline'
        pbar = tqdm(range(num_batches), desc = label)
        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (batch_size, half_len))
            seq = torch.cat((half, half), dim = -1)

            x, labels = seq[:, :-1], seq[:, 1:]

            preds, _ = model(x, return_next_memories = True, ablate_mem = not use_memory)

            loss = F.cross_entropy(
                rearrange(preds, 'b n d -> (b n) d'),
                rearrange(labels, 'b n -> (b n)')
            )

            loss.backward()

            if i % eval_every == 0:
                norms = {}
                for name, param in model.named_parameters():
                    if 'attn_memory' in name and param.grad is not None:
                        key = name.split('.')[-1]
                        if key not in norms:
                            norms[key] = []
                        norms[key].append(param.grad.norm().item())

                if norms:
                    avg_norms = {k: sum(v)/len(v) for k, v in norms.items()}
                    norms_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_norms.items())
                    pbar.write(colored(f"  [Step {i:4d}] Grad Norms  |  {norms_str}", 'dark_grey'))

            optim.step()
            optim.zero_grad()

            loss = loss.item()

            if i % eval_every == 0 or i >= (num_batches - eval_batches):
                model.eval()
                with torch.no_grad():
                    all_preds, _ = model(x, return_next_memories = True, ablate_mem = not use_memory)
                    preds_class = all_preds.argmax(dim = -1)
                    acc = (preds_class[:, half_len:] == labels[:, half_len:]).float().mean()

                    if i >= (num_batches - eval_batches):
                        last_accs.append(acc.item())

                pbar.set_postfix(loss = f'{loss:.3f}', acc = f'{acc.item():.3f}')

        results[label] = sum(last_accs) / len(last_accs)

    # report

    print('')
    print_header()
    for label, acc in results.items():
        print(f'  {label}: {acc:.1%}')
    print_header()

    if not memory_only and 'Baseline' in results:
        memory_acc = results['Memory']
        baseline_acc = results['Baseline']
        advantage = memory_acc - baseline_acc
        if advantage > 0.20:
            print(colored(f'\n  Memory advantage confirmed.', 'green', attrs=['bold']))
        else:
            print(colored(f'\n  No clear advantage.', 'red', attrs=['bold']))

if __name__ == '__main__':
    fire.Fire(train)
