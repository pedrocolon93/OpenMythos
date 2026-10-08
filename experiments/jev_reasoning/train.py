"""
Does the recurrent loop reason? Train a small OpenMythos decision model on
pointer chasing and measure accuracy against hops (difficulty) and n_loops
(compute).

Arms (--arm):
    act     -- the repo's defaults: ACT halting on, 8 loops in training
    fixed   -- ACT off, always 8 loops, backprop through all of them
    random  -- ACT off, loop count sampled per step from a log-normal Poisson
               with mean --mean-loops (Geiping et al., 2025), backprop through
               only the last --grad-loops of them

Training climbs a curriculum (see curriculum()): from a cold start even one
hop sits on a long loss plateau, so it begins with 2 chains of 1 statement
and moves up a level whenever training accuracy holds at --advance-acc. The
last level is the full task with chains of 1..--train-hops statements.
Evaluation goes beyond that in both hops and loops, so the grid shows whether
extra loops are spent on harder problems and whether either axis extrapolates.

    python train.py --arm random --steps 20000 --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Import this checkout's open_mythos, not an installed copy of another branch.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from open_mythos.decision import DecisionModel, decision_loss, decision_probs
from open_mythos.main import MythosConfig
from pointer_task import VOCAB_SIZE, TaskConfig, make_batch, train_batch

EVAL_LOOPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def model_config(args, task: TaskConfig) -> MythosConfig:
    return MythosConfig(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_heads=args.dim // 32,
        n_kv_heads=args.dim // 64,
        max_seq_len=task.max_len + 4 * 8,
        max_loop_iters=32,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=args.dim // 2,
        lora_rank=16,
        rope_theta=10000.0,
        use_act=args.arm == "act",
    )


def curriculum(train_hops: int, final: TaskConfig):
    """
    Levels as (max query hops, task config). Distractor chains grow with the
    query chain so that length never gives the answer's depth away; the last
    level is the full task.
    """
    levels = [
        (1, TaskConfig(n_chains=2, max_distractor_hops=1, noul_frac=final.noul_frac)),
        (1, TaskConfig(n_chains=final.n_chains, max_distractor_hops=2, noul_frac=final.noul_frac)),
    ]
    for k in range(2, train_hops):
        levels.append(
            (k, TaskConfig(n_chains=final.n_chains, max_distractor_hops=k, noul_frac=final.noul_frac))
        )
    levels.append((train_hops, final))
    return levels


def sample_loops(rng: np.random.Generator, mean: float, sigma: float = 0.5) -> int:
    """Log-normal Poisson depth: heavy right tail, so long unrolls are seen too."""
    tau = rng.normal(math.log(mean) - 0.5 * sigma**2, sigma)
    return int(np.clip(rng.poisson(math.exp(tau)) + 1, 1, 32))


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    frac = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


@torch.no_grad()
def evaluate(model, task, device, amp, hops_range, loops_list, per_hop, seed=1234):
    """
    Accuracy and calibration for every (hops, n_loops) cell on a fixed set.

    Choice rows report accuracy and expected calibration error (10 bins on the
    top probability); noul rows report accuracy at threshold 0.5.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    sets = {}
    for h in hops_range:
        noul = np.arange(per_hop) < per_hop // 4
        sets[h] = make_batch(rng, [h] * per_hop, task, noul, device)
    grid = {}
    for n in loops_list:
        for h, b in sets.items():
            with amp:
                logits = model(b["input_ids"], b["slot_mask"], n_loops=n)
            p = decision_probs(logits, b["is_noul"])
            c, y = ~b["is_noul"], b["target"]
            top_p, top_i = p[c].max(-1)
            hit = y[c].gather(-1, top_i[:, None]).squeeze(-1) == 1
            noul_hit = (p[~c].sum(-1) > 0.5) == (y[~c].sum(-1) > 0.5)
            grid[f"{h},{n}"] = {
                "acc": hit.float().mean().item(),
                "ece": ece(top_p, hit),
                "noul_acc": noul_hit.float().mean().item(),
            }
    model.train()
    return grid


def ece(conf: torch.Tensor, hit: torch.Tensor, bins: int = 10) -> float:
    conf, hit = conf.float().cpu(), hit.float().cpu()
    idx = (conf * bins).clamp(max=bins - 1).long()
    total = 0.0
    for i in range(bins):
        m = idx == i
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean() - hit[m].mean()).item()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["act", "fixed", "random"], required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--train-loops", type=int, default=8, help="act/fixed arms")
    ap.add_argument("--mean-loops", type=float, default=8.0, help="random arm")
    ap.add_argument("--grad-loops", type=int, default=8, help="random arm")
    ap.add_argument("--train-hops", type=int, default=8)
    ap.add_argument("--advance-acc", type=float, default=0.9)
    ap.add_argument("--eval-hops", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--per-hop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    name = args.name or f"{args.arm}-s{args.seed}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    task = TaskConfig()
    cfg = model_config(args, task)
    model = DecisionModel(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} params, cfg={cfg}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    cuda = args.device.startswith("cuda")
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cuda
        else torch.autocast("cpu", enabled=False)
    )

    levels = curriculum(args.train_hops, task)
    level, level_steps = 0, [0]
    log, evals = [], []
    t0, run_loss, run_acc = time.time(), 0.0, 0.0
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        max_hops, level_task = levels[level]
        b = train_batch(rng, args.batch_size, max_hops, level_task, args.device)
        if args.arm == "random":
            n, gl = sample_loops(rng, args.mean_loops), args.grad_loops
        else:
            n, gl = args.train_loops, None
        with amp:
            logits = model(b["input_ids"], b["slot_mask"], n_loops=n, grad_loops=gl)
        loss = decision_loss(logits, b["target"], b["is_noul"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            c = ~b["is_noul"]
            run_acc += (
                b["target"][c].gather(-1, logits[c].argmax(-1, keepdim=True)) == 1
            ).float().mean().item()
        run_loss += loss.item()

        if (step + 1) % 100 == 0:
            rec = {
                "step": step + 1,
                "loss": run_loss / 100,
                "train_acc": run_acc / 100,
                "gnorm": gnorm.item(),
                "lr": opt.param_groups[0]["lr"],
                "level": level,
                "elapsed": time.time() - t0,
            }
            log.append(rec)
            if rec["train_acc"] >= args.advance_acc and level + 1 < len(levels):
                level += 1
                level_steps.append(step + 1)
                print(f"level {level} at step {step + 1}: {levels[level]}", flush=True)
            run_loss = run_acc = 0.0
            if (step + 1) % 500 == 0:
                print(json.dumps(rec), flush=True)

        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            g = evaluate(model, task, args.device, amp, [1, 4, 8, 12], [4, 8, 16], 128)
            evals.append({"step": step + 1, "grid": g})
            print(
                f"eval@{step + 1}: "
                + " ".join(f"h{k.replace(',', '/L')}={v['acc']:.2f}" for k, v in g.items()),
                flush=True,
            )

    grid = evaluate(
        model, task, args.device, amp,
        range(1, args.eval_hops + 1), EVAL_LOOPS, args.per_hop,
    )
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out / f"{name}.pt")
    json.dump(
        {
            "name": name,
            "args": vars(args),
            "params": n_params,
            "train_minutes": (time.time() - t0) / 60,
            "log": log,
            "evals": evals,
            "level_steps": level_steps,
            "final_level": level,
            "n_levels": len(levels),
            "grid": grid,
        },
        open(out / f"{name}.json", "w"),
        indent=1,
    )
    print_grid(grid, args.eval_hops)


def print_grid(grid, max_hops):
    print("acc   hops→ " + " ".join(f"{h:>4d}" for h in range(1, max_hops + 1)))
    for n in EVAL_LOOPS:
        row = [grid[f"{h},{n}"]["acc"] for h in range(1, max_hops + 1)]
        print(f"loops {n:>3d}   " + " ".join(f"{a:4.2f}" for a in row))


if __name__ == "__main__":
    main()
