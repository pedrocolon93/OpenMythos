# Renting a GPU

What a pretraining run of this architecture actually costs, and the one thing
worth fixing before paying for any of it.

## Why the parameter count is the wrong number

A dense transformer's training cost is roughly `6N` FLOPs per token, where `N`
is the parameter count. Two features of OpenMythos break that identity in
opposite directions, and both have to be applied before any cost estimate
means anything:

- **The MoE FFN stores more than it uses.** `n_experts` experts are stored;
  `n_experts_per_tok` run. At 64 and 4 that divides the expert bank by 16.
- **The recurrent block uses more than it stores.** One copy of the weights is
  applied `max_loop_iters` times. At T=16 that multiplies it by 16.

The second effect is larger, so these models cost *more* per token than their
parameter count suggests, not less:

| Variant | Stored | Touched per token | Train FLOPs/token |
|---|---|---|---|
| `mythos_300m` | 304M | 394M (T=8) | 2.36 GFLOP |
| `mythos_1b` | 1.06B | 2.58B (T=16) | 15.5 GFLOP |

`mythos_1b` costs 2.4× what a dense 1B costs per token. `training/common.py`
exposes `active_params_per_token` so a run reports the right denominator
instead of the flattering one.

## The number that dominates every other decision

The measured MFU on the dim-768 concept run was **9.8%** — 16.2 TFLOP/s out of
a 4090's 165. The cause is in `MoEFFN`: expert dispatch is a Python loop that
builds one mask per expert, per forward. `mythos_1b` runs 64 experts across 16
loop passes, so **1,024 mask operations per step**, each a separate kernel
launch on a tensor that is mostly zeros.

Nothing else on this page matters as much:

| | 21B tokens (`mythos_1b`) | 100B tokens |
|---|---|---|
| at 35% MFU | 261 H100-hours, ~$700 | 1,242 H100-hours, ~$3,300 |
| at the measured 9.8% | 932 H100-hours, ~$2,400 | 4,437 H100-hours, ~$11,700 |

The gap is **$8,400 on the 100B run**. No provider choice on the market is
worth a quarter of that. Batch the expert dispatch into a single grouped matmul
before renting anything serious.

## Provider rates

Checked 2026-10-18. Rates move; re-check before committing.

| GPU | Modal | Cheapest found | Premium |
|---|---|---|---|
| B200 | $6.25 | $5.98 | 1.0× |
| H200 SXM | $4.54 | $3.59 | 1.3× |
| H100 SXM | $3.95 | $2.69 | 1.5× |
| A100 80GB | $2.50 | $1.19 | 2.1× |
| L40S | $1.95 | $0.79 | 2.5× |

Modal carries a 1.3–2.5× premium on the same silicon, in exchange for
per-second billing and scale-to-zero. For a single long run that elasticity is
worth nothing and the premium is real — rent dedicated hours instead. For a
burst of short paired runs it is worth exactly the idle time it avoids: 71
runs averaging 37 minutes waste about 27 GPU-hours to hourly rounding, which
at H100 rates is $73–$108. Judge it against the premium on the same work.

## The 300M budget run

`modal_300m.py` pretrains `mythos_300m` on FineWeb-Edu inside a fixed credit
budget. It is three commands, and the first two are not optional:

```bash
pip install modal && python3 -m modal setup

modal run modal_300m.py::survey      # ~$1: which card is fastest per dollar
modal run modal_300m.py::calibrate   # ~$0.25: how many tokens the budget buys
modal run --detach modal_300m.py::train
```

`survey` exists because the ranking is not predictable from the spec sheet. If
the model is FLOP-bound the H100 wins outright — 3.2× the arithmetic for 1.6×
the price. If it is launch-overhead-bound, which 9.8% MFU suggests, the extra
arithmetic is unusable and the cheapest card with enough memory wins. The two
hypotheses predict opposite orderings.

`calibrate` exists because the token target sets the cosine schedule, which
must be fixed before the run starts. Size it from a guessed MFU and the run
either stops with its learning rate halfway down the cosine or finishes early
with credit unspent.

### What $30 actually buys

At the measured 9.8% MFU, $26 of credit on a 304M model costing 2.36
GFLOP/token:

| GPU | $/h | hours | tok/s | tokens | % of Chinchilla |
|---|---|---|---|---|---|
| H100 | 3.95 | 6.6 | 41,100 | 0.97B | 16% |
| L40S | 1.95 | 13.3 | 15,000 | 0.72B | 12% |
| A100 80GB | 2.50 | 10.4 | 13,000 | 0.49B | 8% |
| A10 | 1.10 | 23.6 | 5,200 | 0.44B | 7% |

Chinchilla-optimal for 304M is about 6.1B tokens, so a $30 run lands at **7–16%
of it**. That is a real signal and a genuinely undertrained model, and the
report `calibrate` prints says so rather than rounding it up to a headline.

The honest comparison is not against a bigger budget but against the hardware
already on the desk. A 4090 reaches ~6,850 tok/s on this config, so it does the
same 1B tokens in about 40 hours, for free. **What the credit buys is six times
the wall clock, not more tokens.** Spend it when the answer is needed today, or
when the experiment needs more memory than 24GB; otherwise the local card is
strictly cheaper. Where the per-second billing does pay is the thing the single
local GPU cannot do at all: ten paired runs at once, on the Starter plan's 10
GPU concurrency, each stopping and billing nothing the second it finishes.

## Budget mechanics

- **`Budget`** (`training/common.py`) is a wall-clock stop, not a schedule. A
  run that halts on it halts mid-cosine, which costs real loss; it exists so a
  bad throughput estimate costs money instead of costing the run.
- **Modal caps a function at 24h.** The budget is set below the timeout so the
  run stops itself and checkpoints rather than being killed. It resumes from
  the volume either way.
- **The held-out split** is shard 1 of 2 of `sample-10BT`; training takes shard
  0. Splitting the stream rather than the documents is what makes the eval
  number mean something.
- **`mythos_300m` uses GPT-2's 50257-token vocabulary**, not the 201k of
  `openai/gpt-oss-20b` that `MythosTokenizer` defaults to. Embeddings are
  weight-tied, so 201k rows at dim=1024 would spend 206M parameters — two
  thirds of the budget — on a lookup table. It also matches the tokenizer the
  concept tables are built against, so the concept channel can be switched on
  later without rebuilding them.
