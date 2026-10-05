#!/usr/bin/env python3
"""
Pilot experiment: does the ConceptNet concept channel help a small OpenMythos?

Trains tiny OpenMythos models on the same FineWeb-Edu tokens, in the same order,
from the same base initialization, and varies only the concept channel:

  baseline       no concept channel
  real           ConceptNet Numberbatch vectors aligned to the tokenizer
  shuffled_tied  one permutation over the distinct vectors: every token group
                 that shared a vector (" The", " the", " THE") still shares one,
                 but it now carries some other term's vector
  shuffled       rows and span vectors permuted independently, which also breaks
                 those shared-vector groups (a weaker control, kept for contrast)
  shuffled_graph real vectors, but the ConceptNet adjacency is relabeled: every
                 node keeps its degree and every retrieved concept is still a
                 real one, just not a related one. The control for --walk.

Both controls keep coverage, vector statistics and parameter count identical to
"real". shuffled_tied is the one a knowledge claim should be judged by: the
plain shuffle also removes a free "these tokens are the same word" signal that
the real table carries without any semantics, so beating it overstates meaning.

What beating a control can and cannot show. Numberbatch vectors are
distributional word embeddings (word2vec, GloVe) retrofitted with ConceptNet
relations, so a real-over-control gain shows that this pretrained geometry
helps; it cannot say whether the ConceptNet part or the distributional part did
the work. At this size (dim 128, about 12M tokens) pretrained vectors should help
rare, under-trained tokens most, and that kind of gain usually shrinks with scale.

Runs with the same seed start from bit-identical base weights (a concept model
copies the baseline's state_dict; only concept.* parameters are new) and see
batches in the same order, so every comparison is paired by seed.

Held-out loss is also broken down by what the real table injects AT each
position, the same way for every variant including the baseline:

  span_end    a multi-token term ends here, so a span vector is injected here
  unigram     only this token's own single-token row is injected here
  after_span  nothing injected here, but a span ended in the previous 16 tokens
  none        nothing injected here and no span in the previous 16 tokens

These say what is injected at a position, not what can reach it. Once a vector
joins the residual stream, the model's own causal attention can carry it to any
later position under every combiner, so after_span and none are not placebo
positions, and a gain there is not by itself evidence that cross's memory works.

With --walk, a position also retrieves concepts the text does NOT contain, by
walking out from the ones it does (one hop, strongest edges first). Those
retrieved concepts are read by their own attention and added through their own
gate, so they cannot take attention away from the position's own concepts: with
that gate at zero the run IS the run with --walk none. Judge a walking run
against that arm, which isolates what retrieval adds, and against --variant
shuffled_graph, which isolates the content of what is retrieved from its count.

Run:
    python tests/concept_benchmark.py prepare
    python tests/concept_benchmark.py run --variant real --combiner cross --seed 0
    python tests/concept_benchmark.py sweep --seeds 0,1,2 --grid core --jobs 1
    python tests/concept_benchmark.py report
    python tests/concept_benchmark.py plot        # any time, including mid-sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.main import ConceptFusion  # noqa: E402
from tests.small_benchmark import build_tiny_cfg, count_params, fmt_count  # noqa: E402

CATEGORIES = ("span_end", "unigram", "after_span", "none")
# Half precision for the forward and backward pass only. Weights, the optimizer
# and every reported loss stay float32, so a bf16 run is still measured exactly.
AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
AFTER_SPAN_WINDOW = 16
SHUFFLE_SEED = 1234


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TokenChunks(Dataset):
    """Fixed-length (input, target) pairs sliced from one packed token tensor."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data.long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (self.data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def prepare(args: argparse.Namespace) -> None:
    """
    Stream documents, tokenize them, and cache disjoint train and eval tensors.

    The split is by document: training takes whole documents until it has
    enough tokens, and evaluation takes the documents after that, so no
    document contributes to both. An end-of-text token separates documents,
    which also stops a concept span from matching across a document boundary.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    ds = load_dataset(
        args.dataset, name=args.dataset_config or None, split="train", streaming=True
    )

    budgets = {"train": args.train_tokens, "eval": args.eval_tokens}
    bufs: dict[str, list[int]] = {"train": [], "eval": []}
    docs = {"train": 0, "eval": 0}
    current = "train"
    batch: list[str] = []
    t0 = time.perf_counter()

    def flush(texts: list[str]) -> bool:
        nonlocal current
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            bufs[current].extend(ids)
            bufs[current].append(eos)
            docs[current] += 1
            if len(bufs[current]) >= budgets[current]:
                if current == "train":
                    current = "eval"
                else:
                    return True
        return False

    done = False
    for sample in ds:
        text = sample["text"]
        if not text or not text.strip():
            continue
        batch.append(text)
        if len(batch) == 256:
            done = flush(batch)
            batch = []
            n = len(bufs["train"]) + len(bufs["eval"])
            print(f"  {n:,} tokens, {docs['train'] + docs['eval']:,} docs", end="\r", flush=True)
            if done:
                break
    if not done and batch:
        flush(batch)
    print()

    os.makedirs(args.cache_dir, exist_ok=True)
    out = os.path.join(args.cache_dir, "tokens.pt")
    torch.save(
        {
            "train": torch.tensor(bufs["train"], dtype=torch.int32),
            "eval": torch.tensor(bufs["eval"], dtype=torch.int32),
            "meta": {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "tokenizer": args.tokenizer,
                "train_docs": docs["train"],
                "eval_docs": docs["eval"],
            },
        },
        out,
    )
    print(
        f"Wrote {out}: train {len(bufs['train']):,} tokens / {docs['train']:,} docs, "
        f"eval {len(bufs['eval']):,} tokens / {docs['eval']:,} docs "
        f"in {time.perf_counter() - t0:.0f}s"
    )
    sys.stdout.flush()
    # A streaming datasets iterator can hang the interpreter at shutdown while
    # its parquet generator is torn down. Everything is written, so exit hard.
    os._exit(0)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def load_payload(path: str, variant: str = "real", seed: int = 0) -> dict:
    """
    Load the concept table built by scripts/build_concept_table.py, or a control.

    Both controls change only which vector a token or span receives. Which
    positions have a candidate is unchanged, and every vector is still a real
    Numberbatch vector, but it belongs to some other term.

      shuffled       covered unigram rows permuted among themselves and span
                     vectors among themselves, independently
      shuffled_tied  one permutation over the distinct vectors, applied to every
                     row and span that holds one, so exact ties survive

    The permutation seed follows the run seed, so seeds also sample different
    relabelings. Seed 0 keeps the permutation used before that was per seed.
    """
    payload = torch.load(path, map_location="cpu")
    if variant == "real":
        return payload
    g = torch.Generator().manual_seed(SHUFFLE_SEED + seed)
    table = payload["table"].clone()
    covered = (table != 0).any(-1).nonzero(as_tuple=True)[0]
    vectors = payload["span_vectors"]
    if variant == "shuffled":
        table[covered] = payload["table"][covered[torch.randperm(covered.numel(), generator=g)]]
        vectors = vectors[torch.randperm(vectors.shape[0], generator=g)]
        return {**payload, "table": table, "span_vectors": vectors}
    if variant == "shuffled_tied":
        stacked = torch.cat([table[covered], vectors])
        distinct, inverse = torch.unique(stacked, dim=0, return_inverse=True)
        relabeled = distinct[torch.randperm(distinct.shape[0], generator=g)][inverse]
        table[covered] = relabeled[: covered.numel()]
        return {**payload, "table": table, "span_vectors": relabeled[covered.numel():].contiguous()}
    if variant == "shuffled_graph":
        return payload  # the vectors are real; only the graph is broken
    raise ValueError(f"unknown table variant {variant!r}")


def load_graph_payload(path: str, variant: str = "real", seed: int = 0) -> dict:
    """
    Load the ConceptNet graph, or a control that keeps its shape and loses its
    meaning.

    The control relabels node ids inside the adjacency only: every node keeps
    its degree and every edge still points at a real concept with a real
    vector, but at the wrong one. A walk therefore retrieves the same NUMBER of
    concepts, at the same positions, with the same vector statistics, and only
    the relation between a position and what it retrieves is destroyed.

    Without this, a "real" arm would be compared against arms whose graph was
    left untouched, and the comparison would measure nothing about the graph.
    """
    payload = torch.load(path, map_location="cpu")
    if variant == "real":
        return payload
    g = torch.Generator().manual_seed(SHUFFLE_SEED + seed + 7919)
    n_nodes = payload["node_vectors"].shape[0]
    perm = torch.randperm(n_nodes, generator=g)
    return {**payload, "neigh_idx": perm[payload["neigh_idx"].long()].to(torch.int32)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_base_cfg(vocab_size: int, args: argparse.Namespace):
    cfg = build_tiny_cfg(vocab_size, args.seq_len)
    if args.dim != cfg.dim:
        cfg = dataclasses.replace(
            cfg,
            dim=args.dim,
            q_lora_rank=args.dim,
            kv_lora_rank=args.dim // 2,
            expert_dim=args.dim,
        )
    return cfg


def build_model(args: argparse.Namespace, vocab_size: int):
    """Build the run's model, sharing base weights with the baseline for its seed."""
    base_cfg = build_base_cfg(vocab_size, args)
    torch.manual_seed(args.seed)
    base = OpenMythos(base_cfg)
    if args.variant == "baseline":
        return base, base_cfg

    cfg = dataclasses.replace(
        base_cfg,
        use_concept_injection=True,
        concept_dim=300,
        concept_max_span=args.max_span,
        concept_sites=tuple(args.sites.split(",")),
        concept_combiner=args.combiner,
        concept_attn_dim=64,
        concept_gate_init=args.gate_init,
        concept_walk=args.walk,
        concept_walk_k=args.walk_k,
        concept_walk_fanout=args.walk_fanout,
        concept_walk_gate_init=args.walk_gate_init,
    )
    torch.manual_seed(args.seed)
    model = OpenMythos(cfg)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    stray = [k for k in missing if not k.startswith("concept.")]
    if unexpected or stray:
        raise RuntimeError(f"base weights did not transfer: unexpected={unexpected} missing={stray}")
    return model, cfg


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Decay matrices only.

    Gains, norms, the LTI parameters and the per-site concept gates are 1-D.
    Decaying them pulls a gate back toward zero every step, which would bias
    the experiment against the channel.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def categorize(fusion: ConceptFusion, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-position masks for what the real table offers at each position."""
    _, valid = fusion.candidates(x)
    span_end = valid[..., 1:].any(-1)
    has = valid.any(-1)
    T = x.shape[1]
    pos = torch.arange(T, device=x.device).expand_as(span_end)
    last = torch.where(span_end, pos, torch.full_like(pos, -(10**9))).cummax(dim=1).values
    after = ~has & (pos - last <= AFTER_SPAN_WINDOW)
    return {
        "span_end": span_end,
        "unigram": valid[..., 0] & ~span_end,
        "after_span": after,
        "none": ~has & ~after,
    }


@torch.no_grad()
def evaluate(model, loader, device, vocab_size, fusion=None, max_batches=None) -> dict:
    model.eval()
    sums = {c: 0.0 for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    total, n_tok = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.float().view(-1, vocab_size), y.view(-1), reduction="none"
        ).view_as(y)
        total += loss.sum().item()
        n_tok += y.numel()
        if fusion is not None:
            for c, m in categorize(fusion, x).items():
                sums[c] += loss[m].sum().item()
                counts[c] += int(m.sum())
    out = {"loss": total / max(1, n_tok), "tokens": n_tok}
    if fusion is not None:
        out["categories"] = {
            c: {"loss": sums[c] / max(1, counts[c]), "tokens": counts[c]} for c in CATEGORIES
        }
    return out


@torch.no_grad()
def delta_ratios(model, x: torch.Tensor) -> dict:
    """
    Size of each site's concept delta relative to the stream it is added to.

    Measured on one batch by wrapping ConceptFusion.delta. Positions where the
    delta is exactly zero (no candidate reachable) are excluded from "covered".
    """
    records: dict[str, list[tuple[float, float]]] = {}
    fusion = model.concept
    original = fusion.delta

    def spy(site, query, *a, **k):
        out = original(site, query, *a, **k)
        if query is not None:
            q = query.float().norm(dim=-1)
            d = out.float().norm(dim=-1)
            ratio = d / q.clamp_min(1e-12)
            covered = d > 0
            records.setdefault(site, []).append(
                (ratio.mean().item(), ratio[covered].mean().item() if covered.any() else 0.0)
            )
        return out

    fusion.delta = spy
    try:
        model.eval()
        model(x)
    finally:
        del fusion.delta
    return {
        s: {"all": sum(r[0] for r in v) / len(v), "covered": sum(r[1] for r in v) / len(v)}
        for s, v in records.items()
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = torch.load(os.path.join(args.cache_dir, "tokens.pt"))
    vocab_size = 50257 if args.tokenizer == "gpt2" else None
    if vocab_size is None:
        from transformers import AutoTokenizer

        vocab_size = AutoTokenizer.from_pretrained(args.tokenizer).vocab_size

    train_ds = TokenChunks(data["train"], args.seq_len)
    eval_ds = TokenChunks(data["eval"], args.seq_len)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    model, cfg = build_model(args, vocab_size)
    model.to(device)

    # The real table defines the eval categories for every variant, so the
    # baseline is broken down the same way as the concept runs.
    real_payload = load_payload(args.table, "real")
    cat_cfg = dataclasses.replace(
        cfg, use_concept_injection=True, concept_dim=300, concept_max_span=args.max_span,
        concept_sites=("e",), concept_combiner="mean", concept_walk="none",
    )
    fusion = ConceptFusion(cat_cfg).to(device)
    fusion.load(real_payload, tokenizer_id=args.tokenizer)

    load_summary = None
    if args.variant != "baseline":
        payload = real_payload if args.variant == "real" else load_payload(args.table, args.variant, args.seed)
        load_summary = model.load_concept_table(payload, tokenizer_id=args.tokenizer)
    del real_payload

    graph_summary = None
    if args.walk != "none":
        graph_summary = model.load_concept_graph(
            load_graph_payload(args.graph, args.variant, args.seed), tokenizer_id=args.tokenizer
        )
        print(f"[{args.name}] graph {graph_summary}", flush=True)

    n_params = count_params(model)
    concept_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("concept."))
    print(f"[{args.name}] params {fmt_count(n_params)} (concept {concept_params:,})  device={device}", flush=True)

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))

    # Autocast runs the matmuls at half width and keeps a float32 copy of the
    # weights, so memory per token drops and a bigger --batch-size fits. fp16
    # also needs loss scaling, because its exponent range is narrow enough that
    # small gradients flush to zero; bf16 keeps float32's range and does not.
    amp_dtype = AMP_DTYPES[args.precision]
    if amp_dtype is not None and device.type == "cuda" and amp_dtype is torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("this GPU has no bfloat16 support; use --precision fp16 or fp32")
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")

    train_curve, eval_curve = [], []
    data_iter = iter(train_loader)
    timed_tokens, timed_seconds = 0, 0.0
    t_start = time.perf_counter()
    window = []
    for step in range(args.steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)

        t0 = time.perf_counter()
        model.train()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        # The loss itself is always float32: cross entropy over 50k classes is
        # where half precision would actually cost accuracy.
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # clip real gradients, not scaled ones
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss_val = loss.item()  # forces a device sync, so the timing below is honest
        dt = time.perf_counter() - t0
        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step >= 20:
            timed_tokens += x.numel()
            timed_seconds += dt

        window.append(loss_val)
        if (step + 1) % args.log_every == 0:
            avg = sum(window) / len(window)
            window = []
            train_curve.append([step + 1, avg])
            print(f"[{args.name}] step {step + 1:5d}  train {avg:.4f}  lr {lr_at(step, args):.2e}", flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            ev = evaluate(model, eval_loader, device, vocab_size, max_batches=args.eval_batches)
            eval_curve.append([step + 1, ev["loss"]])
            print(f"[{args.name}] step {step + 1:5d}  eval  {ev['loss']:.4f}", flush=True)

    wall = time.perf_counter() - t_start
    # The curve's last point uses the same eval subset as its interim points, so
    # the curve never jumps from switching document sets. The full held-out set
    # is reported separately as final_eval.
    tail = evaluate(model, eval_loader, device, vocab_size, max_batches=args.eval_batches)
    eval_curve.append([args.steps, tail["loss"]])
    print(f"[{args.name}] step {args.steps:5d}  eval  {tail['loss']:.4f}", flush=True)
    final = evaluate(model, eval_loader, device, vocab_size, fusion=fusion)
    print(f"[{args.name}] final eval {final['loss']:.4f} over {final['tokens']:,} tokens", flush=True)

    result = {
        "name": args.name,
        "variant": args.variant,
        "combiner": arm_combiner(args) if args.variant != "baseline" else None,
        "sites": args.sites if args.variant != "baseline" else None,
        "seed": args.seed,
        "tag": args.tag,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dim": cfg.dim,
        "lr": args.lr,
        "tokens_seen": args.steps * args.batch_size * args.seq_len,
        "train_tokens_available": int(data["train"].numel()),
        "params": n_params,
        "concept_params": concept_params,
        "train_curve": train_curve,
        "eval_curve": eval_curve,
        "eval_curve_subset": True,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "max_span": args.max_span,
        "gate_init": args.gate_init,
        "precision": args.precision,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated() / (1 << 30) if device.type == "cuda" else None
        ),
        "walk": args.walk,
        "walk_k": args.walk_k if args.walk != "none" else None,
        "walk_fanout": args.walk_fanout if args.walk != "none" else None,
        "walk_gate_init": args.walk_gate_init if args.walk != "none" else None,
        "graph_summary": graph_summary,
        "shuffle_seed": SHUFFLE_SEED + args.seed if args.variant.startswith("shuffled") else None,
        "final_eval": final,
        "tok_per_sec": timed_tokens / max(1e-9, timed_seconds),
        "wall_sec": wall,
        "load_summary": load_summary,
        "data_meta": data["meta"],
        "device": str(device),
        "torch": torch.__version__,
    }
    if args.variant != "baseline":
        result["gates"] = {
            s: {"l2": g.detach().float().norm().item(), "mean_abs": g.detach().float().abs().mean().item()}
            for s, g in model.concept.gates.items()
        }
        result["span_weight"] = model.concept.span_weight.detach().float().cpu().tolist()
        if args.walk != "none":
            result["walk_gates"] = {
                s: {"l2": g.detach().float().norm().item(),
                    "mean_abs": g.detach().float().abs().mean().item()}
                for s, g in model.concept.walk_gates.items()
            }
        x0, _ = next(iter(eval_loader))
        result["delta_ratio"] = delta_ratios(model, x0.to(device))

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"{args.name}.json")
    with open(out + ".tmp", "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(out + ".tmp", out)
    print(f"[{args.name}] wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# Sweep and report
# ---------------------------------------------------------------------------


CONCEPT_ARMS = [("mean", "e"), ("cross", "e"), ("attend", "embed,e,attn")]
VARIANT_ORDER = ("real", "shuffled_tied", "shuffled", "shuffled_graph")

# Runs are only comparable when these match; report() and sweep() check them.
FINGERPRINT_KEYS = ("steps", "batch_size", "seq_len", "dim", "lr")


def grid(name: str) -> list[tuple]:
    """
    core  baseline, plus every concept arm with the real table and its own
          tie-preserving control, so no arm goes uncontrolled
    full  core, plus the weaker plain shuffle for the two single-site arms
    """
    specs = [("baseline", None, None)]
    for combiner, sites in CONCEPT_ARMS:
        specs.append(("real", combiner, sites))
        specs.append(("shuffled_tied", combiner, sites))
        if name == "full" and sites == "e":
            specs.append(("shuffled", combiner, sites))
    return specs


def arm_combiner(args: argparse.Namespace) -> str:
    """Combiner name as the report should show it, walk included."""
    return args.combiner if args.walk == "none" else f"{args.combiner}_walk{args.walk}"


def run_name(variant, combiner, sites, seed, tag="") -> str:
    base = "baseline" if variant == "baseline" else f"{variant}-{combiner}-{sites.replace(',', '+')}"
    return f"{base}-s{seed}" + (f"-{tag}" if tag else "")


def sweep(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    specs = [(v, c, s, seed, "") for seed in seeds for v, c, s in grid(args.grid)]
    if args.arms:
        # Keep the baseline plus only the named arms, each with its controls.
        wanted = {tuple(a.split("/", 1)) for a in args.arms.split(",")}
        known = {(c, s.replace(",", "+")) for c, s in CONCEPT_ARMS}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown --arms {sorted(unknown)}; choose from {sorted('/'.join(k) for k in known)}")
        specs = [sp for sp in specs if sp[0] == "baseline" or (sp[1], sp[2].replace(",", "+")) in wanted]
    if args.repeat_baseline:
        specs.append(("baseline", None, None, seeds[0], "repeat"))
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)

    shared = [
        "--steps", str(args.steps), "--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len),
        "--dim", str(args.dim), "--lr", str(args.lr), "--warmup", str(args.warmup),
        "--eval-every", str(args.eval_every), "--eval-batches", str(args.eval_batches),
        "--log-every", str(args.log_every), "--device", args.device, "--threads", str(args.threads),
        "--cache-dir", args.cache_dir, "--table", args.table, "--results-dir", args.results_dir,
        "--tokenizer", args.tokenizer, "--max-span", str(args.max_span),
        "--gate-init", str(args.gate_init), "--precision", args.precision,
    ]

    def launch(spec):
        variant, combiner, sites, seed, tag = spec
        name = run_name(variant, combiner, sites, seed, tag)
        done_path = os.path.join(args.results_dir, f"{name}.json")
        if os.path.exists(done_path):
            with open(done_path) as fh:
                done = json.load(fh)
            want = {k: getattr(args, k) for k in FINGERPRINT_KEYS}
            have = {k: done.get(k) for k in FINGERPRINT_KEYS}
            if have != want:
                print(f"CONFLICT {name}: the finished run used {have} but this sweep asks for {want}. "
                      "Use a different --results-dir.", flush=True)
                return name, 1
            print(f"skip {name} (done)", flush=True)
            return name, 0
        cmd = [sys.executable, os.path.abspath(__file__), "run", "--variant", variant,
               "--seed", str(seed), "--name", name, "--tag", tag, *shared]
        if variant != "baseline":
            cmd += ["--combiner", combiner, "--sites", sites]
        log = os.path.join(args.results_dir, "logs", f"{name}.log")
        print(f"start {name}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        print(f"{'done ' if rc == 0 else 'FAIL '} {name} (rc={rc})", flush=True)
        return name, rc

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(launch, specs))
    failed = [n for n, rc in results if rc != 0]
    print(f"\n{len(results) - len(failed)} succeeded, {len(failed)} failed {failed}")


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired(runs_a: dict[int, dict], runs_b: dict[int, dict], pick) -> dict:
    """Seed-paired differences a - b over the seeds both sides actually have."""
    seeds = sorted(set(runs_a) & set(runs_b))
    deltas = [pick(runs_a[s]) - pick(runs_b[s]) for s in seeds]
    m, sd = mean_std(deltas)
    return {"seeds": seeds, "deltas": deltas, "mean": m, "sd": sd}


def verdict(p: dict, noise: float | None) -> str:
    """
    A reading rule, not a significance test.

    Three paired seeds cannot reach significance (a sign test bottoms out at
    p = 0.25 two-sided), so the most this design can say is that an effect is
    consistent in sign and larger than the measured run-to-run noise.
    """
    n = len(p["deltas"])
    if n == 0:
        return "no paired runs"
    if n == 1:
        return "1 pair, indicative only"
    if len({d > 0 for d in p["deltas"]}) > 1:
        return f"signs differ across {n} pairs, inconclusive"
    if noise is not None and abs(p["mean"]) < 2 * abs(noise):
        return f"same sign in {n}/{n}, but within 2x run-to-run noise"
    return f"same sign in {n}/{n}, beyond 2x noise"


def report(args: argparse.Namespace) -> None:
    from collections import Counter

    runs = []
    for f in sorted(os.listdir(args.results_dir)):
        if f.endswith(".json") and f != "report.json":
            with open(os.path.join(args.results_dir, f)) as fh:
                runs.append(json.load(fh))
    if not runs:
        raise SystemExit(f"No results in {args.results_dir}")

    def fingerprint(r):
        return tuple(r.get(k) for k in FINGERPRINT_KEYS)

    main_fp = Counter(fingerprint(r) for r in runs).most_common(1)[0][0]
    excluded = [r["name"] for r in runs if fingerprint(r) != main_fp]
    runs = [r for r in runs if fingerprint(r) == main_fp]

    by_key: dict[str, dict[int, dict]] = {}
    repeats = []
    for r in runs:
        if r.get("tag"):
            repeats.append(r)
            continue
        key = "baseline" if r["variant"] == "baseline" else f"{r['variant']}/{r['combiner']}/{r['sites']}"
        by_key.setdefault(key, {})[r["seed"]] = r
    base = by_key.get("baseline", {})

    noise_deltas = [rep["final_eval"]["loss"] - base[rep["seed"]]["final_eval"]["loss"]
                    for rep in repeats if rep["variant"] == "baseline" and rep["seed"] in base]
    noise = max(noise_deltas, key=abs) if noise_deltas else None

    def sort_key(key):
        if key == "baseline":
            return (0, 0, 0)
        variant, combiner, sites = key.split("/")
        arm = next((i for i, a in enumerate(CONCEPT_ARMS) if a == (combiner, sites)), 9)
        return (1, arm, VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 9)

    loss = lambda r: r["final_eval"]["loss"]  # noqa: E731
    rows = []
    for key in sorted(by_key, key=sort_key):
        seeds = by_key[key]
        row = {
            "key": key,
            "seeds": sorted(seeds),
            "final": mean_std([loss(r) for r in seeds.values()]),
            "tok_per_sec": mean_std([r["tok_per_sec"] for r in seeds.values()]),
            "concept_params": next(iter(seeds.values()))["concept_params"],
        }
        if key != "baseline":
            variant, combiner, sites = key.split("/")
            row["vs_baseline"] = paired(seeds, base, loss)
            row["categories_vs_baseline"] = {
                c: paired(seeds, base, lambda r, c=c: r["final_eval"]["categories"][c]["loss"]) for c in CATEGORIES
            }
            if variant == "real":
                for control in ("shuffled_tied", "shuffled", "shuffled_graph"):
                    other = by_key.get(f"{control}/{combiner}/{sites}")
                    if other:
                        row[f"vs_{control}"] = paired(seeds, other, loss)
            gates, ratios = {}, {}
            for r in seeds.values():
                for site, gv in (r.get("gates") or {}).items():
                    gates.setdefault(site, []).append(gv["mean_abs"])
                for site, rv in (r.get("delta_ratio") or {}).items():
                    ratios.setdefault(site, []).append(rv["covered"])
            row["gate_mean_abs"] = {s: mean_std(v) for s, v in gates.items()}
            row["delta_ratio_covered"] = {s: mean_std(v) for s, v in ratios.items()}
        rows.append(row)

    def cell(p, digits=4):
        if not p or not p["deltas"]:
            return "—"
        per = " ".join(f"{d:+.{digits}f}" for d in p["deltas"])
        return f"{p['mean']:+.{digits}f} [n={len(p['deltas'])}: {per}]"

    example = next(iter(base.values()), runs[0])
    meta = example.get("data_meta", {})
    dataset = meta.get("dataset", "?") + (f" ({meta['dataset_config']})" if meta.get("dataset_config") else "")
    print(f"\n{dataset}: {example['tokens_seen']:,} training tokens per run, dim {example['dim']}, "
          f"{example['steps']} steps, eval over {example['final_eval']['tokens']:,} held-out tokens")
    if excluded:
        print(f"EXCLUDED (different training config from the majority): {', '.join(excluded)}")
    if "categories" in example["final_eval"]:
        cats = example["final_eval"]["categories"]
        tot = sum(v["tokens"] for v in cats.values())
        print("eval positions by what is injected there: " +
              ", ".join(f"{c} {100 * cats[c]['tokens'] / tot:.1f}%" for c in CATEGORIES))
    if noise is not None:
        print(f"run-to-run noise: a same-seed baseline re-run differs by {noise:+.4f}")
    else:
        print("run-to-run noise: unknown (no same-seed baseline repeat)")
    print("All Δ are seed-paired (same base weights, same batches); negative is better. "
          "Verdicts are a reading rule, not significance tests.")

    print("\n| config | seeds | final eval loss | Δ vs baseline | reading |")
    print("|---|---|---|---|---|")
    for row in rows:
        m, sd = row["final"]
        final = f"{m:.4f}" + ("" if math.isnan(sd) else f" ± {sd:.4f}")
        seeds = ",".join(str(s) for s in row["seeds"])
        if row["key"] == "baseline":
            print(f"| {row['key']} | {seeds} | {final} | — | — |")
        else:
            print(f"| {row['key']} | {seeds} | {final} | {cell(row['vs_baseline'])} | {verdict(row['vs_baseline'], noise)} |")

    real_rows = [r for r in rows if r["key"].startswith("real/")]
    if real_rows:
        print("\nDoes the table's content matter? Real table minus its controls (same arm, same seeds)")
        print("| arm | Δ vs tie-preserving shuffle | reading | Δ vs plain shuffle | reading |")
        print("|---|---|---|---|---|")
        for row in real_rows:
            arm = row["key"].split("/", 1)[1]
            tied, plain = row.get("vs_shuffled_tied"), row.get("vs_shuffled")
            print(f"| {arm} | {cell(tied)} | {verdict(tied, noise) if tied else 'control not run'} | "
                  f"{cell(plain)} | {verdict(plain, noise) if plain else 'control not run'} |")
        print("A real-over-control gain shows Numberbatch's pretrained geometry helps; it cannot separate its "
              "ConceptNet relations from its word2vec/GloVe statistics. The attend arm also injects at three "
              "sites, so compare it only with its own control, not with the single-site arms.")

    print("\nΔ loss vs baseline by what is injected at the position (negative = better)")
    print("| config | " + " | ".join(CATEGORIES) + " |")
    print("|---|" + "---|" * len(CATEGORIES))
    for row in rows:
        if row["key"] != "baseline":
            print(f"| {row['key']} | " + " | ".join(
                f"{p['mean']:+.4f}" if p["deltas"] else "—" for p in row["categories_vs_baseline"].values()) + " |")
    print("Every combiner can carry an injected vector forward through attention, so after_span and none are "
          "not placebo positions. Categories also select different kinds of next tokens, so compare a "
          "category's Δ across configs rather than across categories.")

    print("\nchannel usage at end of training (seed means)")
    for row in rows:
        if row["key"] == "baseline":
            continue
        g = ", ".join(f"{s} gate |g| {m:.3f}" for s, (m, _) in row["gate_mean_abs"].items())
        r = ", ".join(f"{s} delta/stream {m:.3f}" for s, (m, _) in row["delta_ratio_covered"].items())
        print(f"  {row['key']}: {g}; {r}; concept params {row['concept_params']:,}")

    print("\nthroughput (tokens/s, seed means). The machine is shared, so treat small differences as noise.")
    for row in rows:
        print(f"  {row['key']}: {row['tok_per_sec'][0]:,.0f}")

    with open(os.path.join(args.results_dir, "report.json"), "w") as fh:
        json.dump({"rows": rows, "noise_repeat_delta": noise, "excluded": excluded,
                   "fingerprint": dict(zip(FINGERPRINT_KEYS, main_fp))}, fh, indent=1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Hue carries the combiner and line style carries the table, so real and its
# controls share a color and differ only in dash. The three hues are
# the first three slots of the dataviz reference palette, validated all-pairs.
SURFACE, INK, INK_2, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COMBINER_COLOR = {"mean": "#2a78d6", "cross": "#eb6834", "attend": "#1baf7a"}


def read_curves(results_dir: str) -> dict[str, dict]:
    """
    Curves for every run, finished or still training.

    A finished run's JSON is authoritative. A run still in progress is read
    from its log, which has train loss every --log-every steps and eval loss at
    each --eval-every checkpoint, so the plot works mid-sweep.
    """
    import re

    step_re = re.compile(r"\] step +(\d+) +(train|eval) +([0-9.]+)")
    final_re = re.compile(r"\] final eval ([0-9.]+)")
    runs: dict[str, dict] = {}
    logs = os.path.join(results_dir, "logs")
    names = {f[:-4] for f in os.listdir(logs) if f.endswith(".log")} if os.path.isdir(logs) else set()
    names |= {f[:-5] for f in os.listdir(results_dir) if f.endswith(".json") and f != "report.json"}
    for name in names:
        js = os.path.join(results_dir, f"{name}.json")
        if os.path.exists(js):
            with open(js) as fh:
                r = json.load(fh)
            evals = r["eval_curve"]
            # Older results ended the curve with the full held-out loss, a
            # different document set from the interim points; drop that point.
            if not r.get("eval_curve_subset") and evals and evals[-1][0] == r["steps"]:
                evals = evals[:-1]
            runs[name] = {"train": r["train_curve"], "eval": evals, "done": True}
            continue
        train, evals = [], []
        with open(os.path.join(logs, f"{name}.log")) as fh:
            for line in fh:
                m = step_re.search(line)
                if m:
                    (train if m.group(2) == "train" else evals).append([int(m.group(1)), float(m.group(3))])
        if train:
            runs[name] = {"train": train, "eval": evals, "done": False}
    return runs


def plot(args: argparse.Namespace) -> None:
    import re

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = read_curves(args.results_dir)
    if not runs:
        raise SystemExit(f"No logs or results in {args.results_dir}")

    name_re = re.compile(r"^(?P<cfg>.+)-s(?P<seed>\d+)(?:-(?P<tag>[a-z]+))?$")
    groups: dict[str, dict[int, dict]] = {}
    repeats: dict[int, dict] = {}
    for name, curves in runs.items():
        m = name_re.match(name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if m.group("tag"):
            if m.group("cfg") == "baseline":
                repeats[seed] = curves
            continue
        groups.setdefault(m.group("cfg"), {})[seed] = curves

    def style(cfg: str) -> dict:
        if cfg == "baseline":
            return {"color": INK_2, "linestyle": "-", "label": "baseline (no concept channel)"}
        variant, combiner, sites = cfg.split("-", 2)
        dash = {"real": "-", "shuffled_tied": (0, (6, 2, 1.5, 2)), "shuffled": (0, (5, 2.5))}
        table = {"real": "real table", "shuffled_tied": "tie-preserving shuffle", "shuffled": "plain shuffle"}
        sites = sites.replace("+", ", ")
        return {
            "color": COMBINER_COLOR.get(combiner, INK_2),
            "linestyle": dash.get(variant, "-"),
            "label": f"{table.get(variant, variant)} · {combiner} · {sites}",
        }

    def order(cfg: str) -> tuple:
        if cfg == "baseline":
            return (0, "", "")
        variant, combiner, sites = cfg.split("-", 2)
        return (1, list(COMBINER_COLOR).index(combiner) if combiner in COMBINER_COLOR else 9,
                VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 9)

    def mean_by_step(curves_by_seed: dict[int, dict], key: str) -> tuple[list[int], list[float]]:
        steps = sorted(set.intersection(*[{s for s, _ in c[key]} for c in curves_by_seed.values()])) \
            if curves_by_seed else []
        vals = [sum(dict(c[key])[s] for c in curves_by_seed.values()) / len(curves_by_seed) for s in steps]
        return steps, vals

    def paired_delta(cfg_runs: dict[int, dict], base_runs: dict[int, dict], key: str):
        seeds = [s for s in cfg_runs if s in base_runs]
        if not seeds:
            return [], []
        per_seed = []
        for s in seeds:
            a, b = dict(cfg_runs[s][key]), dict(base_runs[s][key])
            per_seed.append({st: a[st] - b[st] for st in a if st in b})
        steps = sorted(set.intersection(*[set(d) for d in per_seed]))
        return steps, [sum(d[st] for d in per_seed) / len(per_seed) for st in steps]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    })
    fig, (ax_loss, ax_delta) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28}
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_loss, ax_delta):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)

    base = groups.get("baseline", {})
    marker = dict(marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
    handles, labels = [], []
    lowest, highest = math.inf, -math.inf
    drew_delta = False
    for cfg in sorted(groups, key=order):
        st = style(cfg)
        seeds = groups[cfg]
        steps, vals = mean_by_step(seeds, "train")
        shown = [(s, v) for s, v in zip(steps, vals) if s >= args.skip_steps]
        if not shown:
            continue
        lowest = min(lowest, min(v for _, v in shown))
        highest = max(highest, max(v for _, v in shown))
        (line,) = ax_loss.plot(*zip(*shown), color=st["color"], linestyle=st["linestyle"], linewidth=1.6,
                               solid_capstyle="round", dash_capstyle="round")
        es, ev = mean_by_step(seeds, "eval")
        if es:
            ax_loss.plot(es, ev, color=st["color"], **marker)
        done = all(c["done"] for c in seeds.values())
        last = max(max(s for s, _ in c["train"]) for c in seeds.values())
        seed_note = f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''}"
        handles.append(line)
        labels.append(f"{st['label']}  ({seed_note}{'' if done else f', at step {last}'})")

        if cfg != "baseline":
            ds, dv = paired_delta(seeds, base, "train")
            ds_dv = [(s, v) for s, v in zip(ds, dv) if s >= args.skip_steps]
            if ds_dv:
                drew_delta = True
                ax_delta.plot(*zip(*ds_dv), color=st["color"], linestyle=st["linestyle"], linewidth=1.6,
                              solid_capstyle="round", dash_capstyle="round")
            es, ev = paired_delta(seeds, base, "eval")
            if es:
                drew_delta = True
                ax_delta.plot(es, ev, color=st["color"], **marker)

    rs, rv = paired_delta(repeats, base, "train")
    noise = [(s, v) for s, v in zip(rs, rv) if s >= args.skip_steps]
    if noise:
        (nline,) = ax_delta.plot(*zip(*noise), color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)))
        handles.append(nline)
        labels.append("baseline re-run, same seed (run-to-run noise)")

    ax_delta.axhline(0, color=AXIS, linewidth=1.0)
    ax_loss.set_title("Training loss (line, mean per logged window) and held-out loss (dots)",
                      loc="left", fontsize=11, color=INK, pad=10)
    ax_delta.set_title("Loss minus baseline, same seed and batches (below 0 is better)",
                       loc="left", fontsize=11, color=INK, pad=10)
    ax_loss.set_ylabel("cross-entropy (nats)")
    ax_delta.set_ylabel("Δ nats vs baseline")
    ax_delta.set_xlabel("training step")
    ax_delta.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if math.isfinite(lowest):
        top = min(highest, lowest + args.y_span)
        pad = 0.06 * max(top - lowest, 1e-6)
        ax_loss.set_ylim(lowest - pad, top + pad)
    if drew_delta:
        lo, hi = ax_delta.get_ylim()
        lo, hi = min(lo, -1e-4), max(hi, 1e-4)
        ax_delta.set_ylim(lo, hi)
        # Enough decimals that neighbouring ticks never print the same label.
        step = ax_delta.yaxis.get_major_locator().tick_values(lo, hi)
        spacing = min(abs(b - a) for a, b in zip(step, step[1:])) if len(step) > 1 else hi - lo
        decimals = max(2, int(math.ceil(-math.log10(spacing))) + 1) if spacing > 0 else 3
        ax_delta.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 10 ** -(decimals + 1) else f"{v:+.{decimals}f}")
        )
    else:
        ax_delta.set_yticks([])
        ax_delta.text(0.5, 0.5, "Comparisons appear here once a concept run has logged\n"
                      "past the hidden early steps alongside its same-seed baseline.",
                      transform=ax_delta.transAxes, ha="center", va="center", color=MUTED, fontsize=10)

    leg = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.005), handlelength=2.6)
    for text in leg.get_texts():
        text.set_color(INK_2)
    legend_rows = math.ceil(len(handles) / 2)
    fig.subplots_adjust(bottom=0.08 + 0.03 * legend_rows, top=0.95, left=0.09, right=0.98)

    out = args.out or os.path.join(args.results_dir, "loss.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out} ({len(runs)} runs: " +
          ", ".join(f"{n}{'' if runs[n]['done'] else ' (in progress)'}" for n in sorted(runs)) + ")")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cache-dir", default="data/concept_bench")
        sp.add_argument("--results-dir", default="data/concept_bench/results")
        sp.add_argument("--table", default="data/concept_table_gpt2.pt")
        sp.add_argument("--tokenizer", default="gpt2")

    def training(sp):
        sp.add_argument("--steps", type=int, default=3000)
        sp.add_argument("--batch-size", type=int, default=32)
        sp.add_argument("--seq-len", type=int, default=256)
        sp.add_argument("--dim", type=int, default=128)
        sp.add_argument("--lr", type=float, default=1e-3)
        sp.add_argument("--warmup", type=int, default=200)
        sp.add_argument("--weight-decay", type=float, default=0.1)
        sp.add_argument("--eval-every", type=int, default=500)
        sp.add_argument("--eval-batches", type=int, default=20)
        sp.add_argument("--log-every", type=int, default=100)
        sp.add_argument("--max-span", type=int, default=6)
        sp.add_argument(
            "--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
            help="half precision for the forward/backward pass; weights, optimizer and "
            "every reported loss stay fp32. Changes numerics, so keep one setting across "
            "an experiment rather than mixing arms",
        )
        sp.add_argument("--walk", choices=["none", "fixed"], default="none",
                        help="retrieve concepts the text does not contain by walking the graph")
        sp.add_argument("--graph", default="data/concept_graph_gpt2.pt",
                        help="graph from scripts/build_concept_graph.py; only read when --walk is set")
        sp.add_argument("--walk-k", type=int, default=4, help="retrieved concepts kept per position")
        sp.add_argument("--walk-gate-init", type=float, default=0.0,
                        help="starting value for the retrieval gates; 0 makes the arm start as "
                        "exactly the model with --walk none")
        sp.add_argument("--walk-fanout", type=int, default=4, help="neighbours considered per seed")
        sp.add_argument(
            "--gate-init",
            type=float,
            default=0.0,
            help="starting value for every concept gate; 0 keeps the run baseline-identical "
            "at step 0, a small positive value gives the layers behind the gate a gradient "
            "from the first step",
        )
        sp.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
        sp.add_argument("--threads", type=int, default=2)

    pp = sub.add_parser("prepare", help="stream, tokenize and cache train/eval tokens")
    common(pp)
    pp.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    pp.add_argument("--dataset-config", default="sample-10BT")
    pp.add_argument("--train-tokens", type=int, default=100_000_000)
    pp.add_argument("--eval-tokens", type=int, default=1_000_000)

    rp = sub.add_parser("run", help="train and evaluate one configuration")
    common(rp)
    training(rp)
    rp.add_argument(
        "--variant",
        choices=["baseline", "real", "shuffled", "shuffled_tied", "shuffled_graph"],
        required=True,
    )
    rp.add_argument("--combiner", choices=["mean", "attend", "cross"], default="mean")
    rp.add_argument("--sites", default="e")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--name", default="")
    rp.add_argument("--tag", default="")

    sp = sub.add_parser("sweep", help="run the full grid, a few jobs at a time")
    common(sp)
    training(sp)
    sp.add_argument("--seeds", default="0,1,2")
    sp.add_argument("--jobs", type=int, default=3)
    sp.add_argument("--repeat-baseline", action="store_true")
    sp.add_argument("--arms", default="",
                    help="comma-separated combiner/sites to keep, e.g. mean/e,attend/embed+e+attn "
                    "(the baseline always runs)")
    sp.add_argument("--grid", choices=["core", "full"], default="full",
                    help="core: baseline + each arm with real table and tie-preserving control; "
                    "full: core + plain shuffle for the single-site arms")

    rep = sub.add_parser("report", help="aggregate finished runs into tables")
    common(rep)

    pl = sub.add_parser("plot", help="plot loss curves and paired deltas; works mid-sweep")
    common(pl)
    pl.add_argument("--out", default="", help="PNG path; defaults to <results-dir>/loss.png")
    pl.add_argument("--skip-steps", type=int, default=200,
                    help="hide the steepest early steps so later differences stay visible")
    pl.add_argument("--y-span", type=float, default=2.5,
                    help="height of the loss panel's y range above the lowest plotted loss")

    args = p.parse_args()
    if args.cmd == "run" and not args.name:
        args.name = run_name(args.variant, args.combiner, args.sites, args.seed, args.tag)
    return args


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "run": run, "sweep": sweep, "report": report, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
