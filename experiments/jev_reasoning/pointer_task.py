"""
Pointer chasing as a typed decision, with reasoning depth set by the generator.

The context is a shuffled list of assignments that form several disjoint
chains, each ending in a value:

    x4 = x9 ; x2 = 7 ; x9 = x2 ; x11 = 3 ; ...      (x4 -> x9 -> x2 -> 7)

The question names the head of one chain. A chain of `hops` statements needs
`hops` lookups to resolve. The candidates are the end values of every chain in
the context, so a candidate cannot be ruled out by checking whether it appears
at all: the chain has to be followed. Distractor chains draw their lengths
from the same range as the eval chains, so sequence length does not leak the
answer's depth.

Layout (one token per item):

    BOS  stmt*  Q var ?  (value SLOT)*           -- choice
    BOS  stmt*  Q var ?  value SLOT              -- noul: is var = value?
"""

from dataclasses import dataclass

import numpy as np
import torch

PAD, BOS, EQ, SEMI, Q, QMARK, SLOT = range(7)
N_VARS = 64
N_VALUES = 16
VAR0 = 7
VAL0 = VAR0 + N_VARS
VOCAB_SIZE = VAL0 + N_VALUES


@dataclass
class TaskConfig:
    n_chains: int = 4  # chains in each context, one of them is asked about
    max_distractor_hops: int = 12  # distractor chain lengths ~ U[1, this]
    noul_frac: float = 0.25  # fraction of rows posed as yes/no

    @property
    def max_len(self) -> int:
        stmts = self.n_chains * self.max_distractor_hops * 4
        return 1 + stmts + 3 + 2 * self.n_chains


def make_example(rng: np.random.Generator, hops: int, cfg: TaskConfig, noul: bool):
    """
    Returns (tokens, slot_positions, target_per_slot).
    """
    lengths = [hops] + list(rng.integers(1, cfg.max_distractor_hops + 1, cfg.n_chains - 1))
    need = sum(lengths)
    assert need <= N_VARS, "not enough variables for these chain lengths"
    var_ids = rng.permutation(N_VARS)[:need]
    values = rng.permutation(N_VALUES)[: cfg.n_chains]

    stmts, heads, k = [], [], 0
    for n, val in zip(lengths, values):
        chain = var_ids[k : k + n]  # n variables, n statements; the last binds the value
        k += n
        heads.append(chain[0])
        for a, b in zip(chain[:-1], chain[1:]):
            stmts.append([VAR0 + a, EQ, VAR0 + b, SEMI])
        stmts.append([VAR0 + chain[-1], EQ, VAL0 + val, SEMI])
    order = rng.permutation(len(stmts))

    toks = [BOS]
    for i in order:
        toks += stmts[i]
    toks += [Q, VAR0 + heads[0], QMARK]

    answer = values[0]
    if noul:
        cand = answer if rng.random() < 0.5 else values[rng.integers(1, cfg.n_chains)]
        cands, target = [cand], [float(cand == answer)]
    else:
        cands = list(rng.permutation(values))
        target = [float(c == answer) for c in cands]
    slots = []
    for c in cands:
        toks += [VAL0 + c, SLOT]
        slots.append(len(toks) - 1)
    return toks, slots, target


def make_batch(rng, hops_list, cfg: TaskConfig, noul_flags, device="cpu"):
    """
    Build padded tensors for one batch.

    Returns dict with input_ids (B, L), slot_mask (B, L), target (B, L),
    is_noul (B,), hops (B,)
    """
    rows = [make_example(rng, int(h), cfg, bool(n)) for h, n in zip(hops_list, noul_flags)]
    B, L = len(rows), max(len(r[0]) for r in rows)
    ids = torch.full((B, L), PAD, dtype=torch.long)
    slot = torch.zeros(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, L)
    for i, (t, s, y) in enumerate(rows):
        ids[i, : len(t)] = torch.tensor(t)
        slot[i, s] = True
        tgt[i, s] = torch.tensor(y)
    return {
        "input_ids": ids.to(device),
        "slot_mask": slot.to(device),
        "target": tgt.to(device),
        "is_noul": torch.tensor(np.asarray(noul_flags, dtype=bool)).to(device),
        "hops": torch.tensor(np.asarray(hops_list)).to(device),
    }


def train_batch(rng, batch_size, max_hops, cfg: TaskConfig, device="cpu"):
    hops = rng.integers(1, max_hops + 1, batch_size)
    noul = rng.random(batch_size) < cfg.noul_frac
    return make_batch(rng, hops, cfg, noul, device)


def decode(ids) -> str:
    """Human-readable rendering of one row, for spot checks."""
    names = {PAD: "", BOS: "<s>", EQ: "=", SEMI: ";", Q: "Q", QMARK: "?", SLOT: "[SLOT]"}
    out = []
    for t in ids:
        t = int(t)
        if t in names:
            out.append(names[t])
        elif t < VAL0:
            out.append(f"x{t - VAR0}")
        else:
            out.append(str(t - VAL0))
    return " ".join(w for w in out if w)
