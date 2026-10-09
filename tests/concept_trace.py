#!/usr/bin/env python3
"""
What the concept channel does DURING INFERENCE, token by token.

tests/concept_benchmark.py asks whether the ConceptNet channel helps. This file
asks a different question: at decode time, what does the channel actually
inject, which terms does it come from, and how big is it next to the tensor it
joins?

Two paths.

  train   Trains a small OpenMythos with the channel on (dim 128, combiner
          "attend", sites embed,e,attn, max_span 6) on the same FineWeb-Edu
          tokens the benchmark caches, and SAVES A CHECKPOINT -- which the
          benchmark harness deliberately does not. Config and data conventions
          are the benchmark's, not reimplemented here: build_base_cfg,
          load_token_loaders, load_payload, param_groups, lr_at, evaluate and
          gate_stats are all imported from tests/concept_benchmark.py, and only
          the concept fields are set in this file.

  trace   Loads that checkpoint, runs OpenMythos.generate on prompts chosen to
          exercise the channel, and records per position:
            * the token text
            * which candidate slots were valid, and the ConceptNet term behind
              each one (slot 0 is this token's own unigram row; slot n-1 is the
              length-n term ENDING here, which is what keeps the channel causal)
            * the softmax weight the "attend" combiner put on each candidate,
              per site, and per loop iteration at the "attn" site
            * the per-site gate magnitude
            * ||delta|| / ||stream||, the size of the concept delta relative to
              the tensor it is added to
          then writes JSON and a self-contained HTML view of it.

Nothing here edits the model. The instrumentation wraps ConceptFusion.delta the
way tests/concept_benchmark.py's delta_ratios does, and wraps the model's
forward to learn which absolute positions each decode step covered.

Measuring the ratio honestly. delta_ratios() divides by the norm of the QUERY
handed to delta(). At "embed" and "e" the query IS the tensor the delta is
added to, so the two agree. At "attn" they do not: the query is the recurrent
block's input `combined`, but TransformerBlock.forward adds the delta to
`attn_norm(combined)`. Both are reported, as `ratio` (against the tensor
actually added to) and `ratio_vs_query` (the benchmark's convention), so an
"attn" number here can be read against either.

Naming the terms. data/concept_table_gpt2.pt carries vectors but no term
strings. data/concept_table_gpt2_terms.pt carries the same vectors plus
`unigram_terms` and `span_terms`. trace loads both and refuses to continue
unless they agree bit-exactly on `table`, `span_vectors` and every span index,
so a term name can never be read off a different table than the one the model
is using. Every named slot is then checked against the text it claims to cover:
the decoded token window, lowercased with whitespace joined by underscores,
must equal the term. Mismatches are counted, printed and flagged in the JSON.

Run:
    python tests/concept_trace.py train          # ~20-25 min on mps
    python tests/concept_trace.py trace
    python tests/concept_trace.py html           # re-render the HTML only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from open_mythos import OpenMythos  # noqa: E402
from open_mythos.main import MythosConfig, ngram_keys  # noqa: E402
from tests.small_benchmark import count_params, fmt_count  # noqa: E402
from tests.concept_benchmark import (  # noqa: E402
    build_base_cfg,
    evaluate,
    gate_stats,
    load_payload,
    load_token_loaders,
    lr_at,
    param_groups,
)

# Prompts chosen so the channel has something to do: each contains at least one
# term the table covers only through the SPAN path, i.e. a term the tokenizer
# splits across several tokens, which can therefore only be delivered at its
# last token. `trace --check` verifies that coverage rather than assuming it.
PROMPTS = [
    "Photosynthesis is the process by which plants",
    "The theory of quantum mechanics explains",
    "A neural network learns to recognize",
    "Carbon dioxide and oxygen are exchanged in the",
    "Machine learning and artificial intelligence are",
    "The water cycle moves water through the atmosphere",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_cfg(vocab_size: int, args: argparse.Namespace) -> MythosConfig:
    """The benchmark's base config, with the concept channel switched on."""
    return dataclasses.replace(
        build_base_cfg(vocab_size, args),
        use_concept_injection=True,
        concept_dim=300,
        concept_max_span=args.max_span,
        concept_sites=tuple(args.sites.split(",")),
        concept_combiner=args.combiner,
        concept_attn_dim=64,
        concept_gate_init=args.gate_init,
        concept_walk="none",
    )


def model_fingerprint() -> str:
    """
    sha256 of open_mythos/main.py.

    Several agents work in this tree at once and the model file moves under a
    long run. A checkpoint records the hash of the file it was trained with and
    trace() prints a warning when it no longer matches, so a silent
    architecture change cannot be mistaken for a result.
    """
    import hashlib

    with open(os.path.join(ROOT, "open_mythos", "main.py"), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data, vocab_size, train_loader, eval_loader = load_token_loaders(args)

    cfg = build_cfg(vocab_size, args)
    torch.manual_seed(args.seed)
    model = OpenMythos(cfg)
    model.to(device)
    # Non-persistent buffers: the table must be loaded after construction and
    # after any checkpoint load, which is also why the checkpoint stays small.
    load_summary = model.load_concept_table(
        load_payload(args.table, "real"), tokenizer_id=args.tokenizer
    )

    n_params = count_params(model)
    concept_params = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("concept.")
    )
    print(
        f"[train] params {fmt_count(n_params)} (concept {concept_params:,})  "
        f"device={device}  combiner={cfg.concept_combiner}  sites={cfg.concept_sites}",
        flush=True,
    )
    print(f"[train] table {load_summary}", flush=True)

    opt = torch.optim.AdamW(
        param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )

    train_curve, eval_curve = [], []
    data_iter = iter(train_loader)
    window: list[float] = []
    t_start = time.perf_counter()
    for step in range(args.steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)

        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss at step {step}")

        window.append(loss_val)
        if (step + 1) % args.log_every == 0:
            avg = sum(window) / len(window)
            window = []
            train_curve.append([step + 1, avg])
            elapsed = time.perf_counter() - t_start
            rate = (step + 1) / max(1e-9, elapsed)
            eta = (args.steps - step - 1) / max(1e-9, rate)
            print(
                f"[train] step {step + 1:5d}  train {avg:.4f}  "
                f"lr {lr_at(step, args):.2e}  {1 / rate:.2f}s/step  eta {eta / 60:.1f}min  "
                f"gates "
                + " ".join(
                    f"{s}={v['l2']:.3f}"
                    for s, v in sorted(gate_stats(model.concept.gates).items())
                ),
                flush=True,
            )
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            ev = evaluate(
                model, eval_loader, device, vocab_size, max_batches=args.eval_batches
            )
            eval_curve.append([step + 1, ev["loss"]])
            print(f"[train] step {step + 1:5d}  eval  {ev['loss']:.4f}", flush=True)

    wall = time.perf_counter() - t_start
    final = evaluate(model, eval_loader, device, vocab_size, max_batches=args.eval_batches)
    eval_curve.append([args.steps, final["loss"]])
    print(f"[train] final eval {final['loss']:.4f} in {wall / 60:.1f} min", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)) or ".", exist_ok=True)
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": dataclasses.asdict(cfg),
        "args": vars(args),
        "train_curve": train_curve,
        "eval_curve": eval_curve,
        "final_eval": final,
        "gates": gate_stats(model.concept.gates),
        "span_weight": model.concept.span_weight.detach().float().cpu().tolist(),
        "params": n_params,
        "concept_params": concept_params,
        "load_summary": load_summary,
        "tokens_seen": args.steps * args.batch_size * args.seq_len,
        "wall_sec": wall,
        "data_meta": data["meta"],
        "torch": str(torch.__version__),
        "main_sha": model_fingerprint(),
    }
    torch.save(ckpt, args.ckpt + ".tmp")
    os.replace(args.ckpt + ".tmp", args.ckpt)
    size_mb = os.path.getsize(args.ckpt) / (1 << 20)
    print(f"[train] wrote {args.ckpt} ({size_mb:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Slot naming
# ---------------------------------------------------------------------------


def slot_rows(fusion, input_ids: torch.Tensor, context_ids: torch.Tensor | None):
    """
    Which span_vectors row each candidate slot came from, -1 for none.

    ConceptFusion._candidates returns the vectors and a valid mask but not the
    row behind each hit, and the row is what names the term. This repeats its
    matching -- same buffers, same hash, same searchsorted, same trimming -- to
    recover the rows, and trace() asserts the mask it implies is identical to
    the model's own, so a disagreement cannot pass silently.

    Returns:
        (uni_valid, rows, ids) for the LAST T positions: (B, T) bool for slot 0,
        (B, T, K) int64 rows for slots 1..K-1 (slot 0 is always -1, its term is
        keyed by token id), and the (B, T) token ids those positions hold.
    """
    B, T = input_ids.shape
    ctx = input_ids if context_ids is None else context_ids
    keep = T + fusion.max_span - 1
    if ctx.shape[1] > keep:
        ctx = ctx[:, -keep:]
    L = ctx.shape[1]
    K = fusion.max_span

    rows = torch.full((B, L, K), -1, dtype=torch.int64, device=ctx.device)
    uni_valid = (fusion.concept_table[ctx] != 0).any(-1)

    for n in range(2, K + 1):
        keys = getattr(fusion, f"span_key_{n}")
        if keys.numel() == 0 or L < n:
            continue
        grams = getattr(fusion, f"span_gram_{n}")
        rws = getattr(fusion, f"span_row_{n}")
        win = ctx.unfold(1, n, 1)
        n_windows = win.shape[1]
        flat = win.reshape(-1, n)
        probe = ngram_keys(flat)
        pos = torch.searchsorted(keys, probe).clamp(max=keys.numel() - 1)
        hit = (keys[pos] == probe) & (grams[pos] == flat).all(-1)
        if not bool(hit.any()):
            continue
        sel = hit.nonzero(as_tuple=True)[0]
        batch_idx = torch.div(sel, n_windows, rounding_mode="floor")
        end_idx = sel % n_windows + (n - 1)
        rows[batch_idx, end_idx, n - 1] = rws[pos[sel]]

    return uni_valid[:, -T:], rows[:, -T:], ctx[:, -T:]


def norm_surface(text: str) -> str:
    """The form Numberbatch terms are stored in: lowercase, underscores for gaps."""
    return "_".join(text.strip().lower().split())


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class DeltaSpy:
    """
    Records what ConceptFusion.delta did, per decode step and per site.

    Patches two bound methods on the instances, the way delta_ratios does:
    ConceptFusion.delta, to read the combiner's attention and the delta's size,
    and OpenMythos.forward, to learn the absolute positions each call covered
    (generate passes start_pos, so no guessing is needed).

    The attention weights are recomputed from the same inputs rather than read
    out of the model, so they could drift from what ConceptFusion._read_slots
    actually does. Every call therefore reconstructs the delta from the
    recomputed weights -- gate * proj(sum_k alpha_k * cand_k) -- and compares it
    with the delta the model returned. `recon_err` is the largest absolute
    disagreement, and trace() refuses to write a file where it is not tiny. So
    the alphas in the output are the model's own, or the run fails.

    Extra keyword arguments (loop_t, halted, and anything added later) are
    passed straight through, so a signature change in ConceptFusion.delta does
    not silently drop an argument.
    """

    def __init__(self, model: OpenMythos):
        self.model = model
        self.fusion = model.concept
        self.steps: list[dict] = []
        self._step: dict | None = None
        self._orig_delta = None
        self._orig_forward = None

    def __enter__(self):
        fusion, model = self.fusion, self.model
        self._orig_delta = fusion.delta
        self._orig_forward = model.forward
        attn_norm = model.recurrent.block.attn_norm

        def forward(input_ids, *a, **kw):
            # Positional order after input_ids is (n_loops, kv_cache, start_pos,
            # context_ids); generate() passes them by keyword. Read what is
            # needed and hand everything else straight on.
            start_pos = kw.get("start_pos", a[2] if len(a) > 2 else 0)
            context_ids = kw.get("context_ids", a[3] if len(a) > 3 else None)
            self._step = {
                "start_pos": int(start_pos),
                "ids": input_ids.detach().clone(),
                "context_ids": None if context_ids is None else context_ids.detach().clone(),
                "calls": [],
            }
            self.steps.append(self._step)
            return self._orig_forward(input_ids, *a, **kw)

        def delta(site, query, cand, valid, memory=None, mem_valid=None, ret=None,
                  ret_valid=None, **kw):
            out = self._orig_delta(
                site, query, cand, valid, memory, mem_valid, ret, ret_valid, **kw
            )
            # The combiner's own softmax over this position's slots, recomputed
            # from the same inputs, then checked against the delta it implies.
            q = fusion.queries[site](query)
            k = fusion.key(cand.to(q.dtype))
            scores = (k * q.unsqueeze(2)).sum(-1) * fusion.attn_scale
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            alpha = F.softmax(scores, dim=-1) * valid.any(-1, keepdim=True).to(scores.dtype)
            feat = (alpha.unsqueeze(-1) * cand.to(alpha.dtype)).sum(2)
            recon = fusion.gates[site] * fusion.proj(feat.to(fusion.proj[0].weight.dtype))
            recon_err = (recon - out).abs().max().item()
            # "stream" is the tensor the delta is really added to. At "attn"
            # that is attn_norm(combined), not the query `combined` itself.
            stream = attn_norm(query) if site == "attn" else query
            self._step["calls"].append(
                {
                    "site": site,
                    "loop_t": kw.get("loop_t"),
                    "alpha": alpha[0].float().cpu(),
                    "valid": valid[0].cpu(),
                    "delta_norm": out[0].float().norm(dim=-1).cpu(),
                    "stream_norm": stream[0].float().norm(dim=-1).cpu(),
                    "query_norm": query[0].float().norm(dim=-1).cpu(),
                    "recon_err": recon_err,
                }
            )
            return out

        fusion.delta = delta
        model.forward = forward
        return self

    def __exit__(self, *exc):
        del self.fusion.delta
        del self.model.forward
        return False


def _site_record(call: dict, p: int, slots: list[dict], gates: dict) -> dict:
    """One site's numbers at one position, with alpha aligned to `slots`."""
    alpha = call["alpha"][p]
    d = float(call["delta_norm"][p])
    s = float(call["stream_norm"][p])
    qn = float(call["query_norm"][p])
    return {
        "loop_t": call.get("loop_t"),
        "recon_err": call.get("recon_err"),
        "delta_norm": round(d, 5),
        "stream_norm": round(s, 5),
        "query_norm": round(qn, 5),
        "ratio": round(d / max(s, 1e-12), 5),
        "ratio_vs_query": round(d / max(qn, 1e-12), 5),
        "alpha": [round(float(alpha[sl["slot"]]), 4) for sl in slots],
        "gate_l2": round(gates["l2"], 5),
        "gate_mean_abs": round(gates["mean_abs"], 5),
    }


def trace(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # torch.load defaults to weights_only=True since 2.6. A checkpoint from an
    # earlier revision of this file recorded torch.__version__ as a TorchVersion
    # rather than a str, which that loader rejects; allowlisting the one torch
    # class keeps those checkpoints readable without turning the guard off.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = MythosConfig(**ckpt["cfg"])
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    # Vectors from --table (what the model trains and runs with); names from
    # --terms. Refuse to name anything unless the two tables are the same data.
    payload = load_payload(args.table, "real")
    terms_payload = torch.load(args.terms, map_location="cpu")
    same = torch.equal(payload["table"], terms_payload["table"]) and torch.equal(
        payload["span_vectors"], terms_payload["span_vectors"]
    )
    for n, entry in payload.get("spans", {}).items():
        other = terms_payload.get("spans", {}).get(n)
        same = same and other is not None
        if other is not None:
            same = same and torch.equal(entry["grams"], other["grams"])
            same = same and torch.equal(entry["rows"], other["rows"])
    if not same:
        raise SystemExit(
            f"{args.terms} is not the same table as {args.table}; term names would be "
            "read off different data than the model uses. Rebuild them from one run."
        )
    span_terms = terms_payload["span_terms"]
    unigram_terms = terms_payload["unigram_terms"]
    load_summary = model.load_concept_table(payload, tokenizer_id=args.tokenizer)
    del payload, terms_payload
    print(f"[trace] table {load_summary}", flush=True)

    fusion = model.concept
    gates = gate_stats(model.concept.gates)
    sites = list(fusion.sites)

    # This instrument models one combiner: a softmax over the position's own
    # slots. Anything else would make the alpha column a fiction.
    if fusion.combiner != "attend":
        raise SystemExit(
            f"concept_combiner={fusion.combiner!r}: this trace reports a weight per "
            "candidate slot, which only the 'attend' combiner produces. Train with "
            "--combiner attend."
        )
    if getattr(fusion, "router", "none") != "none":
        raise SystemExit(
            f"concept_router={fusion.router!r}: the delta at 'attn' is then a routed "
            "mixture, which this trace does not model, and the reconstruction check "
            "below would fail. Trace a run with the router off."
        )
    trained_sha = ckpt.get("main_sha")
    now_sha = model_fingerprint()
    if trained_sha and trained_sha != now_sha:
        print(
            f"[trace] WARNING open_mythos/main.py has changed since this checkpoint was "
            f"trained ({trained_sha} -> {now_sha}). The weights still load, but the code "
            f"around them is not the code they were fitted with.",
            flush=True,
        )

    prompts_out = []
    checks = {"slots": 0, "named": 0, "mismatched": 0, "mask_checks": 0,
              "delta_calls": 0, "max_recon_err": 0.0, "examples": []}

    for pi, prompt in enumerate(args.prompts):
        prompt_ids = torch.tensor([tok.encode(prompt)], device=device)
        prompt_len = prompt_ids.shape[1]
        # One token more than is shown: a generated token only gets a concept
        # record once it is fed back in, so the last one displayed needs a step
        # after it. The extra token is traced and then dropped.
        want = args.max_new_tokens
        torch.manual_seed(args.seed + pi)
        with DeltaSpy(model) as spy:
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=want + 1,
                n_loops=args.n_loops,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        ids = out_ids[0].tolist()

        records: dict[int, dict] = {}
        for step in spy.steps:
            T = step["ids"].shape[1]
            start = step["start_pos"]
            uni_valid, rows, pos_ids = slot_rows(fusion, step["ids"], step["context_ids"])
            # The model's own mask for this step, from the unpatched path.
            _, valid = fusion.candidates(step["ids"], step["context_ids"])
            implied = torch.cat([uni_valid.unsqueeze(-1), rows[..., 1:] >= 0], dim=-1)
            if not torch.equal(valid, implied):
                raise RuntimeError(
                    "slot_rows disagrees with ConceptFusion.candidates; term names "
                    "cannot be trusted. Check that both read the same buffers."
                )
            checks["mask_checks"] += 1

            by_site: dict[str, list[dict]] = {s: [] for s in sites}
            for call in step["calls"]:
                by_site[call["site"]].append(call)
                checks["delta_calls"] += 1
                checks["max_recon_err"] = max(checks["max_recon_err"], call["recon_err"])

            for p in range(T):
                abs_pos = start + p
                tid = int(pos_ids[0, p])
                slots = []
                if bool(uni_valid[0, p]):
                    term = unigram_terms[tid]
                    surface = tok.decode([tid])
                    slots.append(
                        {
                            "slot": 0,
                            "span_len": 1,
                            "term": term,
                            "surface": surface,
                            "row": None,
                            "matches": norm_surface(surface) == term,
                        }
                    )
                for n in range(2, fusion.max_span + 1):
                    row = int(rows[0, p, n - 1])
                    if row < 0:
                        continue
                    lo = abs_pos - n + 1
                    surface = tok.decode(ids[lo : abs_pos + 1]) if lo >= 0 else ""
                    term = span_terms[row]
                    slots.append(
                        {
                            "slot": n - 1,
                            "span_len": n,
                            "term": term,
                            "surface": surface,
                            "row": row,
                            "matches": norm_surface(surface) == term,
                        }
                    )
                for sl in slots:
                    checks["slots"] += 1
                    if sl["term"]:
                        checks["named"] += 1
                    if not sl["matches"]:
                        checks["mismatched"] += 1
                        if len(checks["examples"]) < 20:
                            checks["examples"].append(
                                {
                                    "prompt": pi,
                                    "pos": abs_pos,
                                    "span_len": sl["span_len"],
                                    "surface": sl["surface"],
                                    "term": sl["term"],
                                }
                            )

                site_recs = {}
                for s in sites:
                    calls = by_site[s]
                    if not calls:
                        continue
                    per = [_site_record(c, p, slots, gates[s]) for c in calls]
                    if len(per) == 1:
                        site_recs[s] = per[0]
                    else:
                        # "attn" runs once per loop iteration, with a query that
                        # changes with depth, so the combiner can select
                        # differently at each one.
                        mean = {
                            "delta_norm": round(
                                sum(r["delta_norm"] for r in per) / len(per), 5
                            ),
                            "stream_norm": round(
                                sum(r["stream_norm"] for r in per) / len(per), 5
                            ),
                            "query_norm": round(
                                sum(r["query_norm"] for r in per) / len(per), 5
                            ),
                            "ratio": round(sum(r["ratio"] for r in per) / len(per), 5),
                            "ratio_vs_query": round(
                                sum(r["ratio_vs_query"] for r in per) / len(per), 5
                            ),
                            "alpha": [
                                round(sum(r["alpha"][i] for r in per) / len(per), 4)
                                for i in range(len(slots))
                            ],
                            "gate_l2": per[0]["gate_l2"],
                            "gate_mean_abs": per[0]["gate_mean_abs"],
                            "iters": per,
                        }
                        site_recs[s] = mean

                records[abs_pos] = {
                    "pos": abs_pos,
                    "token": tok.decode([tid]),
                    "token_id": tid,
                    "role": "prompt" if abs_pos < prompt_len else "generated",
                    "step": 0 if start == 0 else start - prompt_len + 1,
                    "slots": slots,
                    "sites": site_recs,
                }

        kept = prompt_len + want
        positions = []
        for p in range(kept):
            rec = records.get(p)
            if rec is None:  # cannot happen: every kept position was consumed
                raise RuntimeError(f"no record for position {p}")
            rec["next_token"] = tok.decode([ids[p + 1]]) if p + 1 < len(ids) else None
            positions.append(rec)

        prompts_out.append(
            {
                "prompt": prompt,
                "prompt_len": prompt_len,
                "generated": tok.decode(ids[prompt_len:kept]),
                "text": tok.decode(ids[:kept]),
                "positions": positions,
            }
        )
        n_span = sum(
            1 for r in positions for s in r["slots"] if s["span_len"] > 1
        )
        print(
            f"[trace] {prompt!r} -> {tok.decode(ids[prompt_len:kept])!r}  "
            f"({n_span} span hits over {kept} positions)",
            flush=True,
        )

    # The alphas are recomputed, not read out of the model. If rebuilding the
    # delta from them does not reproduce the delta the model returned, they are
    # not the weights the model used and nothing here should be believed.
    if checks["max_recon_err"] > args.recon_tol:
        raise SystemExit(
            f"reconstructing the delta from the recorded attention weights was off by "
            f"{checks['max_recon_err']:.3e} (tolerance {args.recon_tol:.1e}) over "
            f"{checks['delta_calls']} calls. ConceptFusion.delta no longer matches what "
            "this file recomputes; fix DeltaSpy before trusting a trace."
        )
    checks["max_recon_err"] = float(checks["max_recon_err"])

    out = {
        "meta": {
            "checkpoint": os.path.abspath(args.ckpt),
            "main_sha_trained": trained_sha,
            "main_sha_now": now_sha,
            "table": os.path.abspath(args.table),
            "terms": os.path.abspath(args.terms),
            "tokenizer": args.tokenizer,
            "device": str(device),
            "sites": sites,
            "combiner": fusion.combiner,
            "max_span": fusion.max_span,
            "dim": cfg.dim,
            "n_loops": args.n_loops,
            "max_loop_iters": cfg.max_loop_iters,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "gates": gates,
            "span_weight": ckpt.get("span_weight"),
            "params": ckpt.get("params"),
            "concept_params": ckpt.get("concept_params"),
            "train": {
                "steps": ckpt["args"]["steps"],
                "batch_size": ckpt["args"]["batch_size"],
                "seq_len": ckpt["args"]["seq_len"],
                "lr": ckpt["args"]["lr"],
                "gate_init": ckpt["args"]["gate_init"],
                "tokens_seen": ckpt.get("tokens_seen"),
                "wall_sec": ckpt.get("wall_sec"),
                "final_eval": ckpt.get("final_eval"),
                "train_curve": ckpt.get("train_curve"),
                "eval_curve": ckpt.get("eval_curve"),
            },
            "load_summary": load_summary,
            "checks": checks,
        },
        "prompts": prompts_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[trace] wrote {os.path.abspath(args.json)}", flush=True)
    print(
        f"[trace] self-check: {checks['slots']} slots named, "
        f"{checks['named']} with a term, {checks['mismatched']} where the term does "
        f"not match the text it covers, {checks['mask_checks']} mask comparisons passed, "
        f"{checks['delta_calls']} deltas reconstructed to within "
        f"{checks['max_recon_err']:.2e}",
        flush=True,
    )
    for ex in checks["examples"]:
        print(f"[trace]   MISMATCH {ex}", flush=True)

    render_html(out, args.html)


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------


def render_html(trace_data: dict, path: str) -> None:
    payload = json.dumps(trace_data, separators=(",", ":"))
    # Keep the JSON out of the parser's way: a term could in principle contain
    # "</script>", and an escaped "<" is still valid JSON.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace(
        "\u2029", "\\u2029"
    )
    doc = HTML_TEMPLATE.replace("__TRACE_JSON__", payload)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(doc)
    size_kb = os.path.getsize(path) / 1024
    print(f"[html] wrote {os.path.abspath(path)} ({size_kb:.0f} KB)", flush=True)


def html_cmd(args: argparse.Namespace) -> None:
    with open(args.json) as fh:
        render_html(json.load(fh), args.html)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept Channel Trace</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f6f6f4;
  --panel: #ffffff;
  --panel-2: #f0f0ec;
  --ink: #1a1a18;
  --ink-2: #53524d;
  --ink-3: #85837b;
  --line: #e0ded7;
  --line-2: #c8c6bd;
  --heat-rgb: 26 86 168;        /* size of the delta */
  --heat-ink: #ffffff;
  --span-rgb: 126 40 150;       /* a multi-token term landed here */
  --gen-rgb: 176 92 8;          /* the model wrote this, it was not given */
  --ok: #1b6a3e;
  --warn: #9a2a1f;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  --shadow: 0 1px 2px rgba(0,0,0,.05), 0 10px 26px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141519;
    --panel: #1c1e23;
    --panel-2: #24262c;
    --ink: #eceef2;
    --ink-2: #b3b7bf;
    --ink-3: #858a94;
    --line: #2e313a;
    --line-2: #3d414a;
    --heat-rgb: 110 168 255;
    --heat-ink: #0a0c10;
    --span-rgb: 206 146 255;
    --gen-rgb: 240 176 102;
    --ok: #6fd39a;
    --warn: #ff9b8f;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 26px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --bg: #141519;
  --panel: #1c1e23;
  --panel-2: #24262c;
  --ink: #eceef2;
  --ink-2: #b3b7bf;
  --ink-3: #858a94;
  --line: #2e313a;
  --line-2: #3d414a;
  --heat-rgb: 110 168 255;
  --heat-ink: #0a0c10;
  --span-rgb: 206 146 255;
  --gen-rgb: 240 176 102;
  --ok: #6fd39a;
  --warn: #ff9b8f;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 26px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1060px; margin: 0 auto; padding: 32px 16px 96px; }

header h1 { font-size: 1.62rem; line-height: 1.2; margin: 0 0 8px; letter-spacing: -.015em; }
header p.lede { margin: 0 0 22px; color: var(--ink-2); max-width: 66ch; }
header p.lede code { font-family: var(--mono); font-size: .88em; }
.topbar { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
          justify-content: space-between; margin-bottom: 18px; }

button { font: inherit; color: var(--ink); background: var(--panel);
         border: 1px solid var(--line-2); border-radius: 8px; padding: 5px 11px; cursor: pointer; }
button:hover { background: var(--panel-2); }
button:focus-visible { outline: 2px solid rgb(var(--heat-rgb)); outline-offset: 2px; }
.seg button[aria-pressed="true"] {
  background: rgb(var(--heat-rgb)); border-color: rgb(var(--heat-rgb)); color: var(--heat-ink);
}
.seg { display: inline-flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.seg .lbl { color: var(--ink-3); font-size: .76rem; text-transform: uppercase;
            letter-spacing: .07em; margin-right: 2px; }

.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
        gap: 1px; background: var(--line); border: 1px solid var(--line);
        border-radius: 10px; overflow: hidden; margin: 0 0 14px; }
.meta div { background: var(--panel); padding: 9px 12px; }
.meta dt { color: var(--ink-3); font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; margin: 0; }
.meta dd { margin: 2px 0 0; font-family: var(--mono); font-size: .84rem; }

.note { font-size: .86rem; color: var(--ink-2); background: var(--panel); border: 1px solid var(--line);
        border-left: 3px solid rgb(var(--heat-rgb)); border-radius: 8px; padding: 10px 13px; margin: 0 0 24px; }
.note code { font-family: var(--mono); font-size: .9em; background: var(--panel-2); padding: 1px 4px; border-radius: 4px; }
.note .ok { color: var(--ok); font-weight: 650; }
.note .warn { color: var(--warn); font-weight: 650; }

.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
        box-shadow: var(--shadow); padding: 16px 16px 14px; margin-bottom: 20px; }
.card h2 { font-size: .95rem; margin: 0 0 4px; font-weight: 650; }
.card .sub { font-family: var(--mono); font-size: .8rem; color: var(--ink-3); margin: 0 0 12px; word-break: break-word; }
.card .sub b { color: rgb(var(--gen-rgb)); font-weight: 600; }
.card .peak { font-size: .8rem; color: var(--ink-2); margin: 0 0 12px; }
.card .peak b { font-family: var(--mono); color: var(--ink); font-weight: 600; }

.flow { display: flex; flex-wrap: wrap; gap: 3px 3px; margin-bottom: 8px; }
.tok { --a: 0;
  position: relative; font-family: var(--mono); font-size: .85rem; line-height: 1.3;
  padding: 4px 5px 5px; border-radius: 5px; cursor: pointer;
  border: 1px solid transparent; white-space: pre; color: var(--ink);
  background-color: rgb(var(--heat-rgb) / var(--a));
}
.tok.hot { color: var(--heat-ink); }
.tok.bare { border: 1px dashed var(--line-2); color: var(--ink-3); }
.tok.gen { box-shadow: inset 0 2.5px 0 rgb(var(--gen-rgb)); }
.tok.span::after { content: ""; position: absolute; left: 3px; right: 3px; bottom: 1.5px;
                   height: 2.5px; background: rgb(var(--span-rgb)); border-radius: 2px; }
.tok:hover { border-color: var(--line-2); }
.tok[aria-pressed="true"] { outline: 2px solid rgb(var(--heat-rgb)); outline-offset: 1px; }
.tok .ws { opacity: .32; }

.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
          font-size: .76rem; color: var(--ink-2); margin: 0 0 4px; }
.legend .sw { display: inline-block; width: 36px; height: 10px; border-radius: 3px; vertical-align: -1px;
              margin-right: 6px; border: 1px solid var(--line-2);
              background: linear-gradient(90deg, rgb(var(--heat-rgb) / .04), rgb(var(--heat-rgb))); }
.legend .u, .legend .t { display: inline-block; width: 16px; height: 3px; border-radius: 2px;
                         vertical-align: 3px; margin-right: 6px; }
.legend .u { background: rgb(var(--span-rgb)); }
.legend .t { background: rgb(var(--gen-rgb)); }
.legend .d { display: inline-block; width: 14px; height: 10px; border: 1px dashed var(--line-2);
             border-radius: 3px; vertical-align: -1px; margin-right: 6px; }

.chart { margin: 8px 0 2px; }
.chart svg { display: block; width: 100%; height: auto; }

.detail { border-top: 1px dashed var(--line-2); margin-top: 12px; padding-top: 12px; font-size: .88rem; }
.detail .empty { color: var(--ink-3); margin: 0; }
.detail h3 { font-size: .92rem; margin: 0 0 6px; font-family: var(--mono); font-weight: 650; }
.detail h3 .role { font-family: var(--sans); font-size: .7rem; font-weight: 650; text-transform: uppercase;
                   letter-spacing: .07em; margin-left: 8px; padding: 1px 6px; border-radius: 20px;
                   background: var(--panel-2); color: var(--ink-3); }
.detail h3 .role.gen { color: rgb(var(--gen-rgb)); }
.detail p.next { color: var(--ink-2); font-size: .84rem; margin: 0 0 12px; }
.detail p.next code { font-family: var(--mono); background: var(--panel-2); padding: 1px 5px; border-radius: 4px; }

table.slots { width: 100%; border-collapse: collapse; margin: 0 0 14px; font-size: .83rem; }
table.slots th, table.slots td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }
table.slots th { color: var(--ink-3); font-size: .68rem; text-transform: uppercase; letter-spacing: .07em; font-weight: 650; }
table.slots th.a { text-align: right; }
table.slots td.term { font-family: var(--mono); }
table.slots tr.span td.kind, table.slots tr.span td.term { color: rgb(var(--span-rgb)); }
table.slots td.a { white-space: nowrap; text-align: right; }
table.slots td.a .bar { display: inline-block; width: 46px; height: 7px; background: var(--panel-2);
                        border-radius: 4px; overflow: hidden; vertical-align: 1px; margin-right: 6px; }
table.slots td.a .bar i { display: block; height: 100%; background: rgb(var(--heat-rgb)); border-radius: 4px; }
table.slots td.a span { font-family: var(--mono); }
.chk { font-size: .76rem; }
.chk.y { color: var(--ok); }
.chk.n { color: var(--warn); font-weight: 650; }

.sites { display: grid; grid-template-columns: repeat(auto-fit, minmax(226px, 1fr)); gap: 10px; }
.site { background: var(--panel-2); border: 1px solid var(--line); border-radius: 9px; padding: 10px 11px; }
.site .name { font-family: var(--mono); font-size: .84rem; font-weight: 650; margin-bottom: 2px; }
.site .desc { font-size: .73rem; color: var(--ink-3); margin-bottom: 8px; line-height: 1.4; }
.site .kv { display: flex; justify-content: space-between; gap: 10px; font-size: .79rem; }
.site .kv span:first-child { color: var(--ink-2); }
.site .kv span:last-child { font-family: var(--mono); }
.site .kv.big span:last-child { font-weight: 650; }
.site .iters { margin-top: 8px; padding-top: 7px; border-top: 1px solid var(--line);
               font-size: .72rem; color: var(--ink-2); line-height: 1.5; }
.site .iters code { font-family: var(--mono); }

table.agg { width: 100%; border-collapse: collapse; font-size: .84rem; }
table.agg th, table.agg td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
table.agg th:first-child, table.agg td:first-child { text-align: left; }
table.agg th { color: var(--ink-3); font-size: .68rem; text-transform: uppercase;
               letter-spacing: .07em; font-weight: 650; }
table.agg td { font-family: var(--mono); }
table.agg td:first-child { font-family: var(--sans); color: var(--ink-2); }
table.agg td:first-child b { color: var(--ink); font-weight: 600; }
table.agg tr.tot td { border-bottom: none; }
table.agg tr.tot td:first-child b { color: var(--ink); }

footer { color: var(--ink-3); font-size: .78rem; margin-top: 28px; }
footer code { font-family: var(--mono); }

/* Seven columns do not fit a phone. Let the table scroll rather than clip it:
   every alpha stays reachable at any width. */
.tablewrap { overflow-x: auto; margin: 0 0 14px; -webkit-overflow-scrolling: touch; }
.tablewrap table.slots { margin: 0; min-width: 540px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>What the concept channel injects, token by token</h1>
  <p class="lede">A trace of <code>OpenMythos.generate</code> with the ConceptNet channel on.
  Each chip is one position. Its shade is the size of the concept delta next to the tensor it was
  added to; a purple underline marks a position where a multi-token term ended, which is the only
  place such a term can be delivered without handing the model its own next-token target. Click a
  chip for the terms, the combiner&rsquo;s attention over them, and the per-site numbers.</p>
</header>

<div class="topbar">
  <div class="seg" id="siteseg"><span class="lbl">shade by</span></div>
  <div class="seg"><span class="lbl">theme</span><button id="theme" type="button">auto</button></div>
</div>

<dl class="meta" id="meta"></dl>
<p class="note" id="note"></p>
<section class="card" id="summary"></section>
<div id="cards"></div>

<footer>
  Concept vectors: ConceptNet Numberbatch 19.08 (CC-BY-SA 4.0). Written by
  <code>tests/concept_trace.py</code>.
</footer>
</div>

<script id="trace" type="application/json">__TRACE_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("trace").textContent);
const META = DATA.meta;
const SITES = META.sites;
const SITE_DESC = {
  embed: "added to the token embedding, before the Prelude",
  e: "added to the frozen encoding the loop re-injects every iteration",
  attn: "added to the recurrent attention input, recomputed each iteration"
};
let shadeBy = SITES.includes("embed") ? "embed" : SITES[0];
const selected = new Map();   // prompt index -> selected position

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 3) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d);

/* GPT-2 tokens carry their own leading space and it decides whether a unigram
   row exists at all, so whitespace is drawn rather than collapsed. */
function tokHtml(t) {
  let out = "";
  for (const ch of String(t)) {
    if (ch === " ") out += '<span class="ws">·</span>';
    else if (ch === "\n") out += '<span class="ws">↵</span>';
    else if (ch === "\t") out += '<span class="ws">→</span>';
    else out += esc(ch);
  }
  return out === "" ? '<span class="ws">∅</span>' : out;
}

function ratioAt(rec, site) {
  if (site === "max") {
    let m = 0;
    for (const s of SITES) if (rec.sites[s]) m = Math.max(m, rec.sites[s].ratio);
    return m;
  }
  return rec.sites[site] ? rec.sites[site].ratio : 0;
}

/* Per-prompt, per-site scaling. The three sites differ by more than an order of
   magnitude -- the delta at "attn" joins an RMS-normalised tensor -- so one
   global scale would flatten two of them to nothing. */
function scaleFor(prompt, site) {
  let m = 0;
  for (const r of prompt.positions) m = Math.max(m, ratioAt(r, site));
  return m > 0 ? m : 1;
}

const hasSpan = rec => rec.slots.some(s => s.span_len > 1);

function peakOf(prompt, site) {
  let best = null;
  for (const r of prompt.positions) {
    if (!best || ratioAt(r, site) > ratioAt(best, site)) best = r;
  }
  return best;
}

function renderMeta() {
  const t = META.train || {};
  const spans = Object.entries(META.load_summary.spans)
    .map(([n, c]) => `${n}:${(c / 1000).toFixed(0)}k`).join(" ");
  const rows = [
    ["model", `dim ${META.dim} · ${(META.params / 1e6).toFixed(2)}M params`],
    ["concept params", (META.concept_params || 0).toLocaleString()],
    ["combiner", `${META.combiner} · max_span ${META.max_span}`],
    ["sites", SITES.join(", ")],
    ["gate ℓ₂", SITES.map(s => `${s} ${num(META.gates[s].l2, 2)}`).join("  ")],
    ["trained", `${t.steps} steps · ${((t.tokens_seen || 0) / 1e6).toFixed(1)}M tokens`],
    ["held-out loss", num(t.final_eval && t.final_eval.loss, 4)],
    ["decode", `${META.n_loops} loops · T=${META.temperature} · top-k ${META.top_k}`],
    ["unigram rows", META.load_summary.unigram_rows.toLocaleString()],
    ["span entries", spans]
  ];
  document.getElementById("meta").innerHTML = rows.map(([k, v]) =>
    `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");

  const c = META.checks;
  const clean = c.mismatched === 0;
  let h = `<b>Self-check.</b> Every forward pass was re-matched against `
    + `<code>ConceptFusion.candidates</code>: ${c.mask_checks} comparisons, all agreeing on `
    + `every slot. The attention weights below are recomputed, not read out of the model, so `
    + `each of the ${c.delta_calls} deltas was rebuilt from them and compared with the delta the `
    + `model returned — largest disagreement <code>${Number(c.max_recon_err).toExponential(1)}</code>. `
    + `${c.slots} candidate slots fired across these prompts; `
    + (clean
      ? `<span class="ok">all ${c.named} named terms equal the text they cover</span> `
        + `(the token window, lowercased, gaps joined by <code>_</code>).`
      : `<span class="warn">${c.mismatched} of ${c.named} named terms do not match the text they `
        + `cover.</span>`);
  if (META.main_sha_trained && META.main_sha_now && META.main_sha_trained !== META.main_sha_now) {
    h += ` <span class="warn">open_mythos/main.py changed between training `
      + `(<code>${esc(META.main_sha_trained)}</code>) and this trace `
      + `(<code>${esc(META.main_sha_now)}</code>).</span>`;
  }
  document.getElementById("note").innerHTML = h;
}

/* What the channel does on average, split by what was reachable at a position.
   "span" means a multi-token term ended there, "unigram" only the token's own
   row, "none" that nothing was reachable -- and under `attend` a position with
   no candidate gets exactly zero, which is what the last row should show. */
function renderSummary() {
  const groups = { span: [], unigram: [], none: [] };
  for (const prompt of DATA.prompts) {
    for (const r of prompt.positions) {
      const g = hasSpan(r) ? "span" : (r.slots.length ? "unigram" : "none");
      groups[g].push(r);
    }
  }
  const mean = (rows, site) => rows.length
    ? rows.reduce((a, r) => a + ratioAt(r, site), 0) / rows.length : 0;
  const label = {
    span: ["span ends here", "a multi-token term was delivered at this position"],
    unigram: ["unigram only", "this token's own row, no term ended here"],
    none: ["nothing reachable", "no row, no term: the delta is exactly zero"]
  };
  let h = `<h2>What it contributes, averaged over every traced position</h2>`
    + `<p class="peak">Mean <b>\u2016\u0394\u2016/\u2016stream\u2016</b> by what the table `
    + `offered at the position. The three sites are not comparable in absolute size: the delta at `
    + `<b>attn</b> joins an RMS-normalised tensor whose norm is about \u221adim.</p>`
    + `<table class="agg"><thead><tr><th>positions</th><th>count</th>`
    + SITES.map(x => `<th>${esc(x)}</th>`).join("") + `</tr></thead><tbody>`;
  for (const g of ["span", "unigram", "none"]) {
    h += `<tr><td><b>${label[g][0]}</b> \u2014 ${label[g][1]}</td><td>${groups[g].length}</td>`
      + SITES.map(x => `<td>${mean(groups[g], x).toFixed(4)}</td>`).join("") + `</tr>`;
  }
  const all = [].concat(groups.span, groups.unigram, groups.none);
  h += `<tr class="tot"><td><b>all positions</b></td><td>${all.length}</td>`
    + SITES.map(x => `<td>${mean(all, x).toFixed(4)}</td>`).join("") + `</tr></tbody></table>`;
  document.getElementById("summary").innerHTML = h;
}

function renderSiteSeg() {
  const seg = document.getElementById("siteseg");
  seg.querySelectorAll("button").forEach(b => b.remove());
  for (const s of SITES.concat(["max"])) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s === "max" ? "largest site" : s;
    b.setAttribute("aria-pressed", String(s === shadeBy));
    b.onclick = () => { shadeBy = s; renderSiteSeg(); renderCards(); };
    seg.appendChild(b);
  }
}

function chartSvg(prompt, site) {
  const P = prompt.positions, n = P.length;
  const W = 1000, H = 96, pad = 20, gap = n > 60 ? 0.6 : 2;
  const bw = (W - 2 * pad) / n;
  const scale = scaleFor(prompt, site);
  const floor = H - pad;
  let bars = "";
  for (let i = 0; i < n; i++) {
    const r = ratioAt(P[i], site);
    const h = Math.max(1, (r / scale) * (H - 2 * pad - 12));
    const x = pad + i * bw;
    const gen = P[i].role === "generated";
    bars += `<rect x="${x.toFixed(2)}" y="${(floor - h).toFixed(2)}" `
      + `width="${Math.max(0.8, bw - gap).toFixed(2)}" height="${h.toFixed(2)}" rx="1" `
      + `fill="${gen ? "rgb(var(--gen-rgb))" : "rgb(var(--heat-rgb))"}" opacity="${gen ? 0.9 : 0.62}"></rect>`;
    if (hasSpan(P[i])) {
      bars += `<rect x="${x.toFixed(2)}" y="${(floor + 2).toFixed(2)}" `
        + `width="${Math.max(0.8, bw - gap).toFixed(2)}" height="2.5" rx="1" `
        + `fill="rgb(var(--span-rgb))"></rect>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="size of the concept delta at each position">`
    + `<line x1="${pad}" y1="${floor}" x2="${W - pad}" y2="${floor}" stroke="var(--line-2)" stroke-width="1"></line>`
    + bars
    + `<text x="${pad}" y="${pad - 5}" fill="var(--ink-3)" font-size="11" `
    + `font-family="ui-monospace, Menlo, monospace">`
    + `‖Δ‖/‖stream‖ at ${site === "max" ? "the largest site" : site}`
    + `, peak ${scale.toFixed(4)}</text></svg>`;
}

function detailHtml(prompt, pos) {
  if (pos === undefined || pos === null) {
    return '<p class="empty">Click a token above for its concept slots, the combiner’s '
      + 'attention over them, and each site’s numbers.</p>';
  }
  const rec = prompt.positions.find(r => r.pos === pos);
  if (!rec) return '<p class="empty">no record for that position</p>';
  const gen = rec.role === "generated";

  let h = `<h3>[${rec.pos}] ${tokHtml(rec.token)}`
    + `<span class="role${gen ? " gen" : ""}">${gen ? "generated · step " + rec.step : "prompt"}</span></h3>`;
  if (rec.next_token !== null && rec.next_token !== undefined) {
    h += `<p class="next">This position’s logits produced <code>${tokHtml(rec.next_token)}</code>.</p>`;
  }

  if (!rec.slots.length) {
    h += `<p class="next">Nothing fired here: this token’s unigram row is all zeros and no indexed `
      + `term ends at it. The projection is bias-free, so under <code>attend</code> the delta is `
      + `exactly zero at every site.</p>`;
  } else {
    h += `<div class="tablewrap"><table class="slots"><thead><tr><th>slot</th>`
      + `<th>ConceptNet term</th><th>covers</th>`
      + `<th>term = text</th>`
      + SITES.map(s => `<th class="a">α at ${esc(s)}</th>`).join("")
      + `</tr></thead><tbody>`;
    rec.slots.forEach((sl, i) => {
      h += `<tr class="${sl.span_len > 1 ? "span" : ""}">`
        + `<td class="kind">${sl.slot} <span style="color:var(--ink-3)">`
        + `${sl.span_len === 1 ? "unigram" : "span " + sl.span_len}</span></td>`
        + `<td class="term">${esc(sl.term || "—")}</td>`
        + `<td class="term" style="color:var(--ink-2)">${tokHtml(sl.surface)}</td>`
        + `<td class="chk ${sl.matches ? "y" : "n"}">${sl.matches ? "✓" : "✗ mismatch"}</td>`;
      for (const s of SITES) {
        const a = rec.sites[s] ? rec.sites[s].alpha[i] : 0;
        h += `<td class="a"><span class="bar"><i style="width:${(a * 100).toFixed(1)}%"></i></span>`
          + `<span>${num(a)}</span></td>`;
      }
      h += `</tr>`;
    });
    h += `</tbody></table></div>`;
  }

  h += `<div class="sites">`;
  for (const s of SITES) {
    const r = rec.sites[s];
    if (!r) continue;
    h += `<div class="site"><div class="name">${esc(s)}</div>`
      + `<div class="desc">${esc(SITE_DESC[s] || "")}</div>`
      + `<div class="kv big"><span>‖Δ‖ / ‖stream‖</span><span>${num(r.ratio, 4)}</span></div>`
      + `<div class="kv"><span>‖Δ‖</span><span>${num(r.delta_norm, 4)}</span></div>`
      + `<div class="kv"><span>‖stream‖</span><span>${num(r.stream_norm, 3)}</span></div>`
      + `<div class="kv"><span>gate ℓ₂</span><span>${num(r.gate_l2, 3)}</span></div>`
      + `<div class="kv"><span>mean |gate|</span><span>${num(r.gate_mean_abs, 4)}</span></div>`;
    if (r.iters) {
      h += `<div class="iters">one read per loop iteration: `
        + r.iters.map(it => `<code>${num(it.ratio, 4)}</code>`).join(" ")
        + `<br>against the query <code>combined</code> instead (the benchmark’s `
        + `<code>delta_ratios</code> convention): <code>${num(r.ratio_vs_query, 4)}</code></div>`;
    }
    h += `</div>`;
  }
  return h + `</div>`;
}

function renderCards() {
  const host = document.getElementById("cards");
  host.innerHTML = "";
  DATA.prompts.forEach((prompt, pi) => {
    const scale = scaleFor(prompt, shadeBy);
    const peak = peakOf(prompt, shadeBy);
    const card = document.createElement("section");
    card.className = "card";

    const peakTerms = peak && peak.slots.length
      ? peak.slots.map(s => s.term).join(", ") : "nothing";
    card.insertAdjacentHTML("beforeend",
      `<h2>Prompt ${pi + 1}</h2>`
      + `<p class="sub">${esc(prompt.prompt)}<b>${esc(prompt.generated)}</b></p>`
      + `<p class="peak">Largest delta at <b>${esc(METAsite())}</b>: position `
      + `<b>${peak ? peak.pos : "—"}</b> ${peak ? tokHtml(peak.token) : ""}, `
      + `ratio <b>${peak ? ratioAt(peak, shadeBy).toFixed(4) : "—"}</b>, from <b>${esc(peakTerms)}</b>.</p>`);

    const flow = document.createElement("div");
    flow.className = "flow";
    for (const rec of prompt.positions) {
      const r = ratioAt(rec, shadeBy);
      const a = Math.min(1, r / scale);
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tok"
        + (rec.role === "generated" ? " gen" : "")
        + (hasSpan(rec) ? " span" : "")
        + (rec.slots.length ? "" : " bare")
        + (a > 0.62 ? " hot" : "");
      b.style.setProperty("--a", a.toFixed(3));
      b.innerHTML = tokHtml(rec.token);
      b.title = `[${rec.pos}] ${JSON.stringify(rec.token)}  ratio ${r.toFixed(4)} — `
        + (rec.slots.length ? rec.slots.map(s => s.term || "?").join(", ") : "no concept");
      b.setAttribute("aria-pressed", String(selected.get(pi) === rec.pos));
      b.onclick = () => {
        selected.set(pi, selected.get(pi) === rec.pos ? null : rec.pos);
        renderCards();
      };
      flow.appendChild(b);
    }
    card.appendChild(flow);

    card.insertAdjacentHTML("beforeend",
      `<div class="legend">`
      + `<span><span class="sw"></span>0 → ${scale.toFixed(4)}</span>`
      + `<span><span class="u"></span>a multi-token term ends here</span>`
      + `<span><span class="t"></span>generated, not prompt</span>`
      + `<span><span class="d"></span>no concept reachable</span></div>`
      + `<div class="chart">${chartSvg(prompt, shadeBy)}</div>`);

    const det = document.createElement("div");
    det.className = "detail";
    det.innerHTML = detailHtml(prompt, selected.get(pi));
    card.appendChild(det);
    host.appendChild(card);
  });
}

const METAsite = () => shadeBy === "max" ? "the largest site" : shadeBy;

const themeBtn = document.getElementById("theme");
const THEMES = ["auto", "light", "dark"];
let ti = 0;
themeBtn.onclick = () => {
  ti = (ti + 1) % THEMES.length;
  const t = THEMES[ti];
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
  themeBtn.textContent = t;
};

renderMeta();
renderSummary();
renderSiteSeg();
renderCards();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    # data/ is gitignored, so nothing here can be committed by accident. Point
    # CONCEPT_TRACE_OUT elsewhere (a scratch directory) to keep the repo clean.
    scratch = os.environ.get("CONCEPT_TRACE_OUT", "data/concept_trace")

    def common(sp):
        sp.add_argument("--ckpt", default=os.path.join(scratch, "concept_trace_model.pt"))
        sp.add_argument("--table", default="data/concept_table_gpt2.pt")
        sp.add_argument("--tokenizer", default="gpt2")
        sp.add_argument("--device", default=default_device)
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--seed", type=int, default=0)

    tp = sub.add_parser("train", help="train a small concept model and save a checkpoint")
    common(tp)
    tp.add_argument("--cache-dir", default="data/concept_bench")
    tp.add_argument("--steps", type=int, default=1500)
    tp.add_argument("--batch-size", type=int, default=64)
    tp.add_argument("--seq-len", type=int, default=256)
    tp.add_argument("--dim", type=int, default=128)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--warmup", type=int, default=150)
    tp.add_argument("--weight-decay", type=float, default=0.1)
    tp.add_argument("--eval-every", type=int, default=500)
    tp.add_argument("--eval-batches", type=int, default=10)
    tp.add_argument("--log-every", type=int, default=50)
    tp.add_argument("--max-span", type=int, default=6)
    tp.add_argument("--sites", default="embed,e,attn")
    tp.add_argument("--combiner", choices=["mean", "attend", "cross"], default="attend")
    tp.add_argument("--gate-init", type=float, default=0.0)

    rp = sub.add_parser("trace", help="decode with the channel instrumented, write JSON + HTML")
    common(rp)
    rp.add_argument("--terms", default="data/concept_table_gpt2_terms.pt",
                    help="same table, plus the term string behind every row")
    rp.add_argument("--json", default=os.path.join(scratch, "concept_trace.json"))
    rp.add_argument("--html", default=os.path.join(scratch, "concept_trace.html"))
    rp.add_argument("--max-new-tokens", type=int, default=24)
    rp.add_argument("--n-loops", type=int, default=4)
    rp.add_argument("--temperature", type=float, default=0.8)
    rp.add_argument("--top-k", type=int, default=40)
    rp.add_argument("--recon-tol", type=float, default=1e-5,
                    help="largest tolerated disagreement between the delta the model "
                    "returned and the one rebuilt from the recorded attention weights")
    rp.add_argument("--prompt", dest="prompts", action="append", default=None,
                    help="repeatable; defaults to the built-in set")

    hp = sub.add_parser("html", help="re-render the HTML from an existing trace JSON")
    hp.add_argument("--json", default=os.path.join(scratch, "concept_trace.json"))
    hp.add_argument("--html", default=os.path.join(scratch, "concept_trace.html"))

    args = p.parse_args()
    if args.cmd == "trace" and not args.prompts:
        args.prompts = list(PROMPTS)
    return args


def main() -> None:
    args = parse_args()
    {"train": train, "trace": trace, "html": html_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
