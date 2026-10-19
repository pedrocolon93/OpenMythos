"""
Train the decision model on the easiest pointer task (2 chains, 1 hop) and
watch why it does or does not learn.

    python decision_probe.py --model mythos --variant hop1 --steps 1000 --probe
    python decision_probe.py --model mythos --variant copy --steps 1500
    python decision_probe.py --model vanilla --variant hop1 --steps 15000

--variant copy  puts the answer value itself where the queried variable goes
                (`Q 5 ? 5 [SLOT] 2 [SLOT]`): one lookup, a first-order signal.
--probe         at a few steps, prints per-module gradient norm and distance
                moved from init, how much the slot hidden state moves when the
                context / query / candidate is changed, and the size and
                context-sensitivity of each stage's output (prelude, recurrent,
                coda) before the final norm.

Findings (2026-10-08, dim 128, lr 1e-3):
  copy, 2 loops     learns: loss 0.69 -> 0.15 by step 1500
  hop1, mythos      0.693 (chance) for 10k steps at lr 3e-4/1e-3, 2 or 4 loops
  hop1, vanilla     0.693 for 7k+ steps at 2 and 4 layers: the task, not the arch
  --probe, mythos   parameters move (recurrent ~7.7 from init by step 1000) but
                    slot output sensitivity to any input falls to 0.000, and the
                    recurrent state norm grows 15 -> 261 in an input-independent
                    direction, so the final RMSNorm shrinks gradients ~23x
"""

import argparse
import copy

import numpy as np
import torch

from common import Vanilla, device, small_config
from open_mythos.decision import DecisionModel, decision_loss
from pointer_task import Q, VAL0, VAR0, TaskConfig, train_batch

GROUPS = ["embed", "prelude", "recurrent", "coda", "norm"]


def copy_variant(b):
    """Replace the queried variable with the answer value: a one-lookup task."""
    ids = b["input_ids"]
    rows = torch.arange(len(ids), device=ids.device)
    qpos = (ids == Q).float().argmax(1)
    ans_slot = (b["target"] == 1).float().argmax(1)
    ids[rows, qpos + 1] = ids[rows, ans_slot - 1]


def perturb(ids, what):
    """Change one part of the input, keeping the format valid."""
    x = ids.clone()
    qpos = (x == Q).float().argmax(1)
    rows = torch.arange(len(x), device=x.device)
    pos = torch.arange(x.shape[1], device=x.device)[None].expand_as(x)
    if what == "context":
        m = (x >= VAL0) & (pos < qpos[:, None])
        x[m] = VAL0 + (x[m] - VAL0 + 7) % 16
    elif what == "query":
        x[rows, qpos + 1] = VAR0 + (x[rows, qpos + 1] - VAR0 + 13) % 64
    return x


@torch.no_grad()
def stages(bb, ids, loops):
    """Each stage's output before the final norm, mirroring OpenMythos.hidden_states."""
    T_ = ids.shape[1]
    x = bb.embed(ids)
    f = bb.freqs_cis[:T_]
    mask = bb._causal_mask(T_, ids.device, x.dtype)
    out = {"embed": x}
    for layer in bb.prelude:
        x = layer(x, f, mask)
    out["prelude"] = x
    x = bb.recurrent(x, x, f, mask, loops)
    out["recurrent"] = x
    for layer in bb.coda:
        x = layer(x, f, mask)
    out["coda"] = x
    return out


@torch.no_grad()
def probe(m, init, b, loops):
    named = dict(m.named_parameters())
    for g in GROUPS:
        ps = [(n, p) for n, p in named.items() if n.startswith(f"backbone.{g}")]
        gn = sum(p.grad.norm() ** 2 for _, p in ps if p.grad is not None) ** 0.5
        moved = sum((p - init[n]).norm() ** 2 for n, p in ps) ** 0.5
        print(f"   {g:10s} grad {float(gn):.2e}  moved {float(moved):.2e}")
    ids, slot = b["input_ids"], b["slot_mask"]
    h0 = m.backbone.hidden_states(ids, loops)[slot]
    for what in ("context", "query"):
        h1 = m.backbone.hidden_states(perturb(ids, what), loops)[slot]
        print(f"   slot output moves {(h1 - h0).norm(dim=-1).mean():.3f} when {what} changes")
    a, c = stages(m.backbone, ids, loops), stages(m.backbone, perturb(ids, "context"), loops)
    for k in a:
        h, h2 = a[k][slot], c[k][slot]
        print(
            f"   {k:10s} |h| {h.norm(dim=-1).mean():8.2f}  "
            f"context share {((h - h2).norm(dim=-1) / h.norm(dim=-1)).mean():.2e}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mythos", "vanilla"], default="mythos")
    ap.add_argument("--variant", choices=["hop1", "copy"], default="hop1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4, help="vanilla only")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    dev = device()
    task = TaskConfig(n_chains=2, max_distractor_hops=1, noul_frac=0.0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if args.model == "mythos":
        m = DecisionModel(small_config(task)).to(dev)
    else:
        m = Vanilla(layers=args.layers).to(dev)
    init = copy.deepcopy(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    probe_at = {1, 100, 300, args.steps - 1}

    for s in range(args.steps):
        b = train_batch(rng, 64, 1, task, dev)
        if args.variant == "copy":
            copy_variant(b)
        logits = m(b["input_ids"], b["slot_mask"], n_loops=args.loops)
        loss = decision_loss(logits, b["target"], b["is_noul"])
        opt.zero_grad()
        loss.backward()
        if args.probe and args.model == "mythos" and s in probe_at:
            print(f"step {s} loss {loss.item():.4f}")
            probe(m, init, b, args.loops)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"{args.model} {args.variant} step {s + 1} loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
