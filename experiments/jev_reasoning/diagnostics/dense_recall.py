"""
Dense supervision on the pointer task: after the statements, ask about every
chain head and train next-token cross-entropy on each answer value, instead of
one decision bit per example.

    python dense_recall.py --model vanilla --chains 8 --hops 1 --format compact

--format verbose   `x4 = x9 ; x9 = 7 ;` statements, `Q x4 ? 7 ;` queries
--format compact   `x4 x9 x9 7` statements, `x4 7` queries: key next to value,
                   the standard associative-recall layout

Findings (2026-10-08, verbose format, 6000 steps):
  vanilla, 8 chains, 1 hop     acc 0.34    mythos, 8 chains, 1 hop    acc 0.34
  mythos, 4 chains, <=4 hops   acc 0.52
These are exactly the accuracies of guessing among the values not yet
answered ((1/8)(1/8+...+1/1) = 0.34, (1/4)(1/4+1/3+1/2+1) = 0.52): both
models learn that elimination shortcut and never the lookup itself.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import Vanilla, device, small_config
from open_mythos.main import OpenMythos
from pointer_task import BOS, EQ, PAD, Q, QMARK, SEMI, VAL0, VAR0, TaskConfig


def example(rng, chains, max_hops, compact):
    lengths = rng.integers(1, max_hops + 1, chains)
    vids = rng.permutation(64)[: lengths.sum()]
    vals = rng.permutation(16)[:chains]
    stmts, heads, k = [], [], 0
    for n, v in zip(lengths, vals):
        ch = vids[k : k + n]
        k += n
        heads.append((ch[0], v))
        for a, b in zip(ch[:-1], ch[1:]):
            stmts.append([VAR0 + a, VAR0 + b] if compact else [VAR0 + a, EQ, VAR0 + b, SEMI])
        stmts.append([VAR0 + ch[-1], VAL0 + v] if compact else [VAR0 + ch[-1], EQ, VAL0 + v, SEMI])
    toks = [BOS] + [t for i in rng.permutation(len(stmts)) for t in stmts[i]]
    tgt = [-100] * len(toks)
    for i in rng.permutation(chains):
        h, v = heads[i]
        if compact:  # predict v right after the key
            toks += [VAR0 + h, VAL0 + v]
            tgt += [VAL0 + v, -100]
        else:  # predict v at the "?" position
            toks += [Q, VAR0 + h, QMARK, VAL0 + v, SEMI]
            tgt += [-100, -100, VAL0 + v, -100, -100]
    return toks, tgt


def batch(rng, B, chains, max_hops, compact, dev):
    rows = [example(rng, chains, max_hops, compact) for _ in range(B)]
    L = max(len(r[0]) for r in rows)
    ids = torch.full((B, L), PAD)
    y = torch.full((B, L), -100)
    for i, (t, g) in enumerate(rows):
        ids[i, : len(t)] = torch.tensor(t)
        y[i, : len(g)] = torch.tensor(g)
    return ids.to(dev), y.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mythos", "vanilla"], default="vanilla")
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--format", choices=["verbose", "compact"], default="compact")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--loops", type=int, default=4)
    args = ap.parse_args()

    dev = device()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if args.model == "vanilla":
        m = Vanilla(out="lm").to(dev)
    else:
        m = OpenMythos(small_config(TaskConfig(n_chains=args.chains, max_distractor_hops=args.hops))).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    tag = f"{args.model} {args.format} chains={args.chains} hops<={args.hops}"
    acc = 0.0
    for s in range(args.steps):
        ids, y = batch(rng, 64, args.chains, args.hops, args.format == "compact", dev)
        logits = m(ids, n_loops=args.loops)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        k = y != -100
        acc += (logits.argmax(-1)[k] == y[k]).float().mean().item()
        if (s + 1) % 500 == 0:
            print(f"{tag} step {s + 1} loss {loss.item():.3f} acc {acc / 500:.3f}", flush=True)
            acc = 0.0


if __name__ == "__main__":
    main()
