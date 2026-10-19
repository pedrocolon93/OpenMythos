#!/usr/bin/env python3
"""
Pretrain OpenMythos 300M on a rented Modal GPU, under a fixed credit budget.

Modal bills per second with no idle charge, which is the reason to use it for
this rather than a cheaper hourly box: a run that stops itself on its budget
stops costing money at that second.

    pip install modal
    python3 -m modal setup                      # one-time browser login

    modal run modal_300m.py::survey             # ~$1, 10 min: which GPU to rent
    modal run modal_300m.py::calibrate          # ~$0.25: how many tokens fit
    modal run --detach modal_300m.py::train     # the real run

Run `survey` before deciding the hardware and `calibrate` before deciding the
token budget. Neither is ceremony:

  * `survey` exists because this model's bottleneck is not known to be
    arithmetic. The 9.8% MFU the sibling dim-768 run measured was traced to
    the MoE dispatch loop and fixed (`moe_dispatch="batched"`, 11.7x end to
    end on MPS; see docs/renting-a-gpu.md), but that fix has never been
    measured on CUDA. If it carries over, the model is FLOP-bound and the
    biggest card wins; if a different launch-overhead wall waits on CUDA, the
    cheapest card with enough memory wins instead. Those predict opposite
    orderings, and five minutes per card settles it.
  * `calibrate` exists because the token target sets the cosine schedule and
    so must be fixed before the run starts. Size it from a guessed MFU and
    the run either stops with its learning rate halfway down the cosine, or
    finishes hours early with credit unspent.

`--detach` on the real run matters: without it, closing the laptop kills the
app. Watch it with `modal app logs <app-id>` (printed on launch) or at
modal.com/apps, and pull the result down with
`modal volume get openmythos-300m-ckpts <file>`.
"""

import modal

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

# Modal list price in $/hour, and the microbatch that fits each card's memory.
# 304M parameters plus fp32 Adam moments is ~3.7GB before activations; eight
# retained loop passes are what actually sets the microbatch. Prices checked
# against modal.com/pricing on 2026-10-18; if one is stale the only effect is
# that the logged spend drifts from the invoice.
HARDWARE = {
    "H100":      (3.95, 16),
    "A100-80GB": (2.50, 16),
    "A100-40GB": (2.10, 8),
    "L40S":      (1.95, 8),
    "A10":       (1.10, 4),
    "L4":        (0.80, 4),
}

# What `train` rents. Change this to whatever `survey` ranks first.
#
# A100-80GB is the default because it is the safe one — 80GB cannot OOM at
# this size — but on paper it is the worst value in the table above. Holding
# MFU constant at the 9.8% the sibling dim-768 run measured, $26 buys:
#
#     H100       6.6h   41,100 tok/s   0.97B tokens
#     L40S      13.3h   15,000 tok/s   0.72B
#     A100-80GB 10.4h   13,000 tok/s   0.49B
#     A10       23.6h    5,200 tok/s   0.44B
#
# The H100 is 3.2x the arithmetic for 1.6x the price, so if this model is
# FLOP-bound the H100 wins outright. If it is launch-overhead-bound the extra
# arithmetic is unusable and the cheapest card with enough memory wins. The
# dispatch fix should have moved it into the first camp — but that was measured
# on MPS, not CUDA, so `survey` is what confirms which camp it is actually in.
# The rows above were computed at the OLD 9.8% MFU and are now pessimistic.
GPU = "A100-80GB"

# What is actually available to spend. The Starter plan's monthly credit is
# $30; the margin covers the survey, the calibration, the image build's CPU
# minutes and the volume.
CREDIT_BUDGET = 26.00

# Set from the number `calibrate` prints. The default is deliberately
# pessimistic rather than optimistic: a target that is too small finishes
# early with a fully annealed learning rate and unspent credit, which is
# recoverable, while one that is too large gets guillotined mid-cosine, which
# is not.
TARGET_TOKENS = 400_000_000

# Modal caps a single function at 24h. The budget is kept below the timeout
# so the run stops itself and checkpoints, rather than being killed.
TIMEOUT_HOURS = 12

# Chinchilla-optimal tokens for 304M stored parameters, for the honesty line
# in the calibration report.
CHINCHILLA_TOKENS = 6.1e9


# ---------------------------------------------------------------------------
# Image and storage
# ---------------------------------------------------------------------------

# Every version here is one the training script was actually exercised
# against locally, not a plausible-looking pin. `datasets` matters most:
# the loader calls `IterableDataset.shard(num_shards=, index=)`, which is
# what splits the stream into the training and held-out halves, and a
# version that renamed or dropped it would fail only after the GPU is rented
# and the image is built.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.6.2",
        "datasets==4.8.5",
        "loguru==0.7.3",
        "hf_transfer==0.1.9",
    )
    .env(
        {
            # Streaming FineWeb-Edu is the one part of this that can be
            # network-bound rather than GPU-bound, and paying by the second to
            # wait on HTTP is an expensive way to discover that.
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/hf",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("open_mythos")
    .add_local_dir("training", remote_path="/root/training")
)

ckpt_vol = modal.Volume.from_name("openmythos-300m-ckpts", create_if_missing=True)
cache_vol = modal.Volume.from_name("openmythos-hf-cache", create_if_missing=True)

app = modal.App("openmythos-300m")

VOLUMES = {"/ckpts": ckpt_vol, "/cache": cache_vol}


# ---------------------------------------------------------------------------
# Running the training script
# ---------------------------------------------------------------------------


def _run(target_tokens: int, max_hours: float, rate: float, micro_batch: int,
         ckpt_dir: str) -> dict:
    """
    Import the training script and run it with the budget wired in.

    Imported by path rather than by name because the module name begins with a
    digit, which `import` cannot express. The environment variables are that
    script's documented outside knobs; every other setting of the run is a
    literal in `training/300m_fine_web_edu.py`, so a run's configuration can
    be read off one file.

    Args:
        target_tokens -- token budget, which also sets the cosine schedule
        max_hours     -- wall-clock stop
        rate          -- $/hour, for the spend figures in the log
        micro_batch   -- sequences per forward; set by the card's memory
        ckpt_dir      -- directory on the mounted volume to checkpoint into

    Returns:
        The training script's summary dict.
    """
    import importlib.util
    import os
    import sys

    os.environ["MYTHOS_TARGET_TOKENS"] = str(target_tokens)
    os.environ["MYTHOS_MAX_HOURS"] = str(max_hours)
    os.environ["MYTHOS_RATE_PER_HR"] = str(rate)
    os.environ["MYTHOS_MICRO_BATCH"] = str(micro_batch)
    os.environ["MYTHOS_CKPT_DIR"] = ckpt_dir

    sys.path.insert(0, "/root/training")
    spec = importlib.util.spec_from_file_location(
        "mythos_300m_train", "/root/training/300m_fine_web_edu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=25 * 60)
def measure(gpu_name: str = GPU, minutes: float = 6.0) -> dict:
    """
    Time the real training loop on whatever GPU this container was given.

    Runs a slice of the actual loop — same model, same batch shape, same
    streaming data — rather than a synthetic benchmark, because the thing most
    likely to be slow here is not the matmuls. A FLOP microbenchmark would
    miss the Python expert-dispatch loop entirely and report a figure several
    times too high.

    `gpu_name` is what this reports and prices itself as; the hardware itself
    comes from the decorator or from `with_options(gpu=...)` at the call site.
    The two are set together by `survey` and must not be set apart, or the
    report prices the wrong card.

    Args:
        gpu_name -- key into HARDWARE, for the price and the microbatch
        minutes  -- wall clock to spend measuring

    Returns:
        The summary dict, plus `gpu`, `rate` and `tokens_per_dollar`.
    """
    rate, micro_batch = HARDWARE[gpu_name]
    summary = _run(
        # Large enough that the measurement is never the thing that stops the
        # run — `minutes` should be. Warmup is 100 steps, so this leaves it.
        target_tokens=4_000_000_000,
        max_hours=minutes / 60.0,
        rate=rate,
        micro_batch=micro_batch,
        # A throwaway directory, so a measurement never becomes something the
        # real run resumes from.
        ckpt_dir=f"/ckpts/measure-{gpu_name}",
    )
    summary["gpu"] = gpu_name
    summary["rate"] = rate
    summary["micro_batch"] = micro_batch
    summary["tokens_per_dollar"] = summary["tok_per_sec"] * 3600.0 / rate
    return summary


@app.function(image=image, gpu=GPU, volumes=VOLUMES,
              timeout=TIMEOUT_HOURS * 60 * 60)
def train() -> dict:
    """
    The real run. Resumes from `/ckpts` if an earlier container left anything.

    Stops on whichever comes first: the token target, the wall-clock budget,
    or Modal's function timeout. The first two write a checkpoint on the way
    out; the third does not, which is why the budget is set below the timeout
    and why the script checkpoints every 500 steps.
    """
    rate, micro_batch = HARDWARE[GPU]
    summary = _run(
        target_tokens=TARGET_TOKENS,
        max_hours=min(CREDIT_BUDGET / rate, TIMEOUT_HOURS - 0.25),
        rate=rate,
        micro_batch=micro_batch,
        ckpt_dir="/ckpts",
    )
    ckpt_vol.commit()

    print()
    print("=" * 72)
    print(f"  {GPU} at ${rate:.2f}/h")
    for key in ("steps", "tokens", "tok_per_sec", "tflops", "mfu",
                "eval_loss", "hours", "stop_reason"):
        print(f"  {key:>14}: {summary[key]}")
    print(f"  {'est. spend':>14}: ${summary['hours'] * rate:.2f}")
    print("=" * 72)
    return summary


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def survey(gpus: str = "H100,A100-80GB,A10", minutes: float = 5.0):
    """
    Time the loop on several cards and rank them by tokens per dollar.

    Sequential, not parallel: these share one HF cache volume, and the point
    is a clean throughput number per card rather than finishing sooner. Three
    cards at five minutes costs well under a dollar, against a $26 decision.

    The default three span the hypothesis: the most arithmetic per second,
    the card with the most memory, and a cheap one. If throughput tracks price
    the model is FLOP-bound and the H100 wins; if the three come back within
    spitting distance of each other it is overhead-bound and the A10 wins by
    costing a third as much per hour.

    A card whose memory cannot hold the configured microbatch fails here with
    a CUDA OOM rather than silently training something smaller. That is the
    intended outcome — it is a real answer about that card.
    """
    results = []
    for name in [g.strip() for g in gpus.split(",") if g.strip()]:
        if name not in HARDWARE:
            print(f"  {name}: not in HARDWARE, skipping")
            continue
        print(f"\n--- measuring {name} ({minutes:.0f} min) ---")
        try:
            results.append(
                measure.with_options(gpu=name).remote(gpu_name=name, minutes=minutes)
            )
        except Exception as exc:  # OOM, capacity, or anything else per-card
            print(f"  {name} failed: {type(exc).__name__}: {exc}")

    if not results:
        print("\nNothing measured.")
        return

    results.sort(key=lambda r: -r["tokens_per_dollar"])
    print()
    print("=" * 78)
    print(f"  {'GPU':<12} {'$/h':>6} {'mb':>3} {'tok/s':>9} {'TFLOP/s':>8} "
          f"{'MFU':>6} {'tok/$':>10} {'tokens for $' + str(int(CREDIT_BUDGET)):>16}")
    for r in results:
        mfu = f"{100 * r['mfu']:.1f}%" if r["mfu"] else "   —"
        buys = r["tokens_per_dollar"] * CREDIT_BUDGET
        print(f"  {r['gpu']:<12} {r['rate']:>6.2f} {r['micro_batch']:>3} "
              f"{r['tok_per_sec']:>9,.0f} {r['tflops']:>8.1f} {mfu:>6} "
              f"{r['tokens_per_dollar'] / 1e6:>9.1f}M {buys / 1e9:>15.2f}B")
    best = results[0]
    print()
    print(f"  Best tokens per dollar: {best['gpu']}")
    print(f"    Set GPU = \"{best['gpu']}\" and run calibrate.")
    print("=" * 78)


@app.local_entrypoint()
def calibrate(minutes: float = 6.0):
    """
    Measure the chosen GPU and print the TARGET_TOKENS the budget affords.

    The printed target is 10% under the measurement so the cosine schedule
    finishes inside the budget rather than being cut off by it.
    """
    summary = measure.remote(gpu_name=GPU, minutes=minutes)
    rate = summary["rate"]
    hours = CREDIT_BUDGET / rate
    affordable = summary["tok_per_sec"] * hours * 3600.0
    target = int(affordable * 0.9 / 1e7) * 10_000_000
    mfu = summary["mfu"]

    print()
    print("=" * 72)
    print(f"  Measured on {summary['device']}  ({GPU} @ ${rate:.2f}/h)")
    print(f"    {summary['tok_per_sec']:>14,.0f} tok/s")
    print(f"    {summary['tflops']:>14.1f} TFLOP/s"
          + (f"   ({100 * mfu:.1f}% MFU)" if mfu else ""))
    print(f"    {summary['gflops_per_tok']:>14.2f} GFLOP/token (6N)")
    print()
    print(f"  ${CREDIT_BUDGET:.2f} buys {hours:.1f}h = {affordable / 1e9:.2f}B tokens")
    print(f"  Chinchilla-optimal for 304M is {CHINCHILLA_TOKENS / 1e9:.1f}B, so this")
    print(f"  run is undertrained by {CHINCHILLA_TOKENS / max(affordable, 1):.1f}x. "
          "Expected at this budget.")
    print()
    print(f"  Set TARGET_TOKENS = {target:_}")
    print("=" * 72)


@app.local_entrypoint()
def main():
    """`modal run modal_300m.py` with no function named runs the real thing."""
    train.remote()
