#!/usr/bin/env python3
"""
OpenMythos 300M pretraining on FineWeb-Edu, sized for a rented single GPU.

The sibling `3b_fine_web_edu.py` assumes a cluster and a token budget set by
how much data you want to spend. This one assumes one GPU and a budget set by
how much money you have, which changes three things:

  * It stops on a wall clock (`MAX_HOURS`), not only on a step count, and it
    writes a checkpoint on the way out so the next container resumes.
  * It reports MFU every log interval. On a rented GPU that is the only number
    that distinguishes training from paying for idle silicon — a measured 10%
    means the same token budget costs three and a half times what a healthy
    35% would.
  * It holds out a disjoint shard of the stream and reports eval loss, because
    a run you cannot judge is not worth renting a GPU for.

Hyperparameters are literal constants below, as in the 3B script. The five
that a rented run has to be able to set from outside — the token target, the
wall clock, the GPU's price, the microbatch and where checkpoints live — read
an environment variable and fall back to the constant; `modal_300m.py` sets
them, and nothing else does. The microbatch is on that list only because the
right value is a property of the rented card's memory, which the script cannot
know: 16 fits an 80GB A100, a 24GB L4 needs 4.

Single GPU:
    python training/300m_fine_web_edu.py

On Modal (recommended; see modal_300m.py for the budget arithmetic):
    modal run modal_300m.py::calibrate
    modal run --detach modal_300m.py::train
"""

import os
import sys
import time
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from contextlib import nullcontext

from open_mythos import OpenMythos
from open_mythos.variants import mythos_300m
from open_mythos.tokenizer import MythosTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    Budget,
    FineWebEduDataset,
    _list_ckpts,
    active_params_per_token,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Hardware peak throughput, for MFU
# ---------------------------------------------------------------------------

# Dense bf16/fp16 tensor-core peak, TFLOP/s, no sparsity. Vendor sparsity
# figures are double these and are not reachable by dense training, so using
# them would halve every MFU number reported here.
PEAK_TFLOPS = {
    "A100": 312.0,
    "H100": 989.0,
    "H200": 989.0,
    "L40S": 362.0,
    "A10": 125.0,
    "L4": 121.0,
    "4090": 165.0,
    "B200": 2250.0,
}


def peak_tflops(device_name: str) -> float | None:
    """
    Dense bf16 peak for the attached GPU, or None if its name is unrecognized.

    Matched by substring because `torch.cuda.get_device_name` returns strings
    like "NVIDIA A100-SXM4-80GB" and "NVIDIA GeForce RTX 4090". An unknown
    name returns None rather than a guess: a wrong denominator makes MFU worse
    than no MFU, since it reads as a hardware fact.

    Args:
        device_name -- what torch reports for the device

    Returns:
        Peak dense TFLOP/s, or None.
    """
    for key, value in PEAK_TFLOPS.items():
        if key in device_name:
            return value
    return None


# ---------------------------------------------------------------------------
# Held-out loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader_iter, device, vocab_size, amp_ctx, batches: int) -> float:
    """
    Mean cross-entropy over `batches` microbatches of the held-out shard.

    The iterator is passed in and kept across calls rather than rebuilt, so
    successive evals walk forward through the held-out stream instead of
    re-scoring the same first few thousand tokens all run. That makes the
    numbers noisier between evals and honest across them; a fixed eval slice
    would look smoother and would start reporting memorization of the slice
    once the run is long enough to come back around.

    Args:
        model       -- the model, left in whatever mode it arrived in
        loader_iter -- long-lived iterator over the held-out DataLoader
        device      -- device to move batches to
        vocab_size  -- for the logits reshape
        amp_ctx     -- autocast context to score under (match training)
        batches     -- how many microbatches to average over

    Returns:
        Mean loss in nats per token.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = next(loader_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
            total += nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1)
            ).item()
    if was_training:
        model.train()
    return total / batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Pretrain the 300M variant on one GPU under a wall-clock budget.

    Differs from the 3B script's lifecycle in that there is no distributed
    init and no FSDP: one GPU holds 304M parameters, their fp32 Adam moments
    and eight loop passes of activations inside 80GB without sharding, and
    FSDP on a single device costs the all-gather without buying anything.

    Lifecycle:
        1. Build the GPT-2 tokenizer and check it against the variant's vocab.
        2. Construct the model, measure active parameters per token for MFU.
        3. Build fused AdamW, resume from the newest checkpoint if present.
        4. Split the stream: shard 0 trains, shard 1 is held out.
        5. Loop until the token target or the wall clock runs out, logging
           loss, MFU and spend, checkpointing periodically and on exit.

    Returns:
        A summary dict. `tok_per_sec` is the median of the per-log-interval
        rates with the first interval dropped — the first interval carries
        cuDNN autotuning, the dataset's first HTTP range requests and the
        allocator reaching steady state, and including it understates
        throughput by enough to matter when the number is then used to size a
        12-hour budget. `calibrate` in modal_300m.py reads exactly this.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 1024
    micro_batch = int(os.environ.get("MYTHOS_MICRO_BATCH", 16))
    # Held fixed at 131,072 tokens per optimizer step regardless of microbatch,
    # so a card that needs a smaller microbatch trains the same schedule more
    # slowly rather than training a different schedule.
    grad_accum = max(1, 131_072 // (micro_batch * seq_len))
    global_batch_tok = micro_batch * grad_accum * seq_len
    lr = 6e-4
    wd = 0.1
    log_every = 20
    eval_every = 250
    eval_batches = 16
    ckpt_every = 500
    dataset_subset = "sample-10BT"

    # The rest of what a rented run sets from outside. TARGET_TOKENS decides the
    # cosine schedule, so it has to be chosen before the run starts rather
    # than discovered during it: `modal_300m.py::calibrate` measures
    # throughput on the real GPU and prints the value that fits the budget.
    target_tokens = int(os.environ.get("MYTHOS_TARGET_TOKENS", 1_500_000_000))
    max_hours = os.environ.get("MYTHOS_MAX_HOURS")
    max_hours = float(max_hours) if max_hours else None
    rate_per_hr = float(os.environ.get("MYTHOS_RATE_PER_HR", 0.0))
    ckpt_dir = os.environ.get("MYTHOS_CKPT_DIR", "checkpoints")

    total_steps = max(1, target_tokens // global_batch_tok)
    warmup_steps = max(100, int(0.02 * total_steps))

    budget = Budget(max_hours, rate_per_hr)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    cfg = mythos_300m()
    encoding = MythosTokenizer("gpt2")
    if encoding.vocab_size != cfg.vocab_size:
        # Not a warning: a mismatch here silently trains a model whose
        # embedding rows and token ids disagree past the smaller of the two.
        raise RuntimeError(
            f"Tokenizer vocab {encoding.vocab_size} != config vocab "
            f"{cfg.vocab_size}. mythos_300m() is built for GPT-2's 50257; "
            "change both together or not at all."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device == "cuda"
        else nullcontext()
    )

    model = OpenMythos(cfg)
    active_per_tok = active_params_per_token(model, cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    gflops_per_tok = 6 * active_per_tok / 1e9
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    peak = peak_tflops(dev_name)

    logger.info(f"Device: {dev_name}  |  AMP dtype: {amp_dtype}")
    logger.info(
        f"Parameters: {n_params:,} stored  |  {active_per_tok:,} touched per token "
        f"(T={cfg.max_loop_iters}, {cfg.n_experts_per_tok}/{cfg.n_experts} experts)"
    )
    logger.info(
        f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
        f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,} | "
        f"target_tokens={target_tokens:,}"
    )
    logger.info(
        f"Training cost: {gflops_per_tok:.2f} GFLOP/token (6N) | "
        f"budget: {budget.max_hours or 'unlimited'}h @ ${rate_per_hr:.2f}/h"
    )
    if peak is None:
        logger.warning(f"No peak-FLOPS entry for {dev_name!r}; MFU will not be reported")

    # ------------------------------------------------------------------
    # Optimizer + resume
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    start_step = 0
    existing = _list_ckpts(ckpt_dir)
    if existing:
        logger.info(f"Resuming from {existing[-1]}")
        start_step = load_checkpoint(model, optimizer, existing[-1], ddp=False)
        logger.success(f"Resumed at step {start_step} of {total_steps}")

    # ------------------------------------------------------------------
    # Data: shard 0 trains, shard 1 is held out
    # ------------------------------------------------------------------
    # Splitting the stream in half rather than by document means the held-out
    # tokens come from documents the training shard never sees, which is the
    # property an eval number needs. It costs half of sample-10BT's 10B
    # tokens, which is irrelevant at a budget measured in single-digit
    # billions.
    train_ds = FineWebEduDataset(encoding, seq_len, dataset_subset, 0, 2)
    eval_ds = FineWebEduDataset(encoding, seq_len, dataset_subset, 1, 2)
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, num_workers=4, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=micro_batch, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    data_iter = iter(train_loader)
    eval_iter = iter(eval_loader)
    step = start_step
    last_saved = start_step
    t0 = time.perf_counter()
    stop_reason = "reached the token target"
    rates: list[float] = []

    while step < total_steps:
        if budget.exhausted():
            stop_reason = f"ran out of wall clock ({budget.summary()})"
            break

        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            rates.append(tok_per_sec)
            mfu = ""
            if peak is not None:
                achieved = tok_per_sec * gflops_per_tok / 1e3  # TFLOP/s
                mfu = f"| {achieved:.1f} TFLOP/s ({100 * achieved / peak:.1f}% MFU) "
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec:,.0f} tok/s {mfu}"
                f"| {step * global_batch_tok / 1e9:.3f}B tokens | {budget.summary()}"
            )
            t0 = time.perf_counter()

        if step % eval_every == 0:
            eval_loss = evaluate(
                model, eval_iter, device, cfg.vocab_size, amp_ctx, eval_batches
            )
            logger.success(
                f"step {step:6d} | eval loss {eval_loss:.4f} "
                f"| eval ppl {torch.tensor(eval_loss).exp():.2f}"
            )
            t0 = time.perf_counter()  # don't bill eval time to training throughput

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, cfg.vocab_size, ckpt_dir,
                ddp=False, master=True,
            )
            last_saved = step
            t0 = time.perf_counter()

    if step != last_saved:
        # Without this the tokens since the last periodic save are lost, which
        # for a budget-stopped run is most of the final interval.
        save_checkpoint(
            model, optimizer, step, cfg, cfg.vocab_size, ckpt_dir,
            ddp=False, master=True,
        )

    final = evaluate(model, eval_iter, device, cfg.vocab_size, amp_ctx, eval_batches)
    logger.success(
        f"Stopped: {stop_reason}. step {step}/{total_steps}, "
        f"{step * global_batch_tok / 1e9:.3f}B tokens, eval loss {final:.4f} "
        f"(ppl {torch.tensor(final).exp():.2f}), {budget.summary()}"
    )

    steady = sorted(rates[1:]) or sorted(rates)
    median_rate = steady[len(steady) // 2] if steady else 0.0
    achieved = median_rate * gflops_per_tok / 1e3
    return {
        "device": dev_name,
        "steps": step,
        "tokens": step * global_batch_tok,
        "tok_per_sec": median_rate,
        "tflops": achieved,
        "mfu": (achieved / peak) if peak else None,
        "gflops_per_tok": gflops_per_tok,
        "eval_loss": final,
        "hours": budget.elapsed_hours,
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    main()
