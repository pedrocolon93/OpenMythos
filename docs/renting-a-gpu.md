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
a 4090's 165. At 35% MFU the same work costs a third as much:

| | 21B tokens (`mythos_1b`) | 100B tokens |
|---|---|---|
| at 35% MFU | 261 H100-hours, ~$700 | 1,242 H100-hours, ~$3,300 |
| at 9.8% MFU | 932 H100-hours, ~$2,400 | 4,437 H100-hours, ~$11,700 |

The gap is **$8,400 on the 100B run**, which is more than any provider choice
on the market is worth. So it is worth knowing exactly where it went.

### Where the time went

Expert dispatch was a Python loop over `(top-k slot, expert)` pairs. For a
64-expert, top-4 config that is 256 iterations per `MoEFFN` call, and the
recurrent block calls it once per loop iteration — so **2,048 iterations per
forward pass at T=8, and 4,096 at T=16**. Three measurements, all on MPS:

**1. Time scaled with expert count at constant arithmetic.** Holding top-4
fixed so every row does the same 6.44 GFLOP of expert work:

| n_experts | ms/call | vs 8 experts |
|---|---|---|
| 8 | 111 | 1.00× |
| 16 | 242 | 2.18× |
| 32 | 434 | 3.91× |
| 64 | 764 | 6.89× |

Eight times the experts, 6.9 times the time, identical FLOPs. Whatever was
costing the time, it was not arithmetic.

**2. 18% of the call was spent before any expert ran.** `mask.any()` forces a
device-to-host synchronization every iteration, and `flat[mask]` has to count
selected rows on the host, so it synchronizes too:

| statement | share of the call |
|---|---|
| `mask = expert_ids == eid` | 1.2% |
| `mask.any()` (sync) | 5.0% |
| `flat[mask]` (bool gather) | 12.1% |
| expert matmuls + `out[mask] +=` | 81.6% |

**3. The matmuls were launch-bound, not compute-bound.** Each expert saw only
`n_tokens × topk / n_experts` rows, far too few to fill a GPU. The same 2.15
GFLOP, split different ways:

| | GFLOP/s |
|---|---|
| 1 matmul, (4096×512)@(512×512) | 1,929 |
| 64 matmuls of (64×512)@(512×512) | 677 |
| 256 matmuls of (16×512)@(512×512) | 249 |
| 1 `bmm`, (64,64,512)@(64,512,512) | 3,440 |

So the 81.6% was launch latency wearing a matmul's clothing, and `bmm` was
5× faster than the loop it replaces.

### The fix

`moe_dispatch="batched"` (now the default) sorts the `n_tokens × topk`
assignments by expert, pads each expert's group out to the largest group, and
runs every expert in three `bmm` calls. No Python loop, and one sync per call
instead of 256. Capacity is the largest observed group rather than a fixed
factor, so **no token is dropped — this is exact, not approximate**. Measured
padding waste at 64 experts is 1.36×, which the `router_bias` load balancer is
what keeps small.

| n_experts | loop | batched | speedup |
|---|---|---|---|
| 8 | 133 ms | 26.7 ms | 4.99× |
| 16 | 172 ms | 24.5 ms | 7.04× |
| 32 | 386 ms | 29.5 ms | 13.1× |
| 64 | 724 ms | 32.7 ms | **22.1×** |

End to end on a `mythos_300m`-shaped model, forward and backward:
**5,444 ms/step → 465 ms/step, an 11.7× speedup.** The batched MoE reaches 10%
of this machine's single-large-matmul throughput, against 0.5% for the loop.

`moe_dispatch="loop"` keeps the original path. The two are not bit-identical —
they sum the same products in a different order — but they agree to 1.2e-16 in
fp64 and one float epsilon in fp32, with gradients matching to 3.3e-16.
Checkpoints written with the old per-expert layout load into the new one
through a state-dict hook, so the billion-token run's weights are not stranded.

**Caveat on these numbers:** all of them were measured on Apple MPS, because
the 4090 box was unreachable when this was written. The direction is not in
doubt — three independent measurements agree, and the mechanism is understood —
but the *magnitude* on CUDA will differ, since CUDA's per-launch cost and sync
cost are both different from Metal's. The run's own MFU line is the number to
trust. Re-measure before quoting 11.7× as a CUDA figure.

### Expert granularity is a throughput decision too

Holding total expert parameters at 50.3M *and* active parameters per token
fixed, only changing how finely the bank is divided:

| n_experts | expert_dim | top-k | batched | loop |
|---|---|---|---|---|
| 64 | 512 | 4 | 35.3 ms | 672 ms |
| 32 | 1024 | 2 | 11.2 ms | 139 ms |
| 16 | 2048 | 1 | **7.9 ms** | 52 ms |

Same weights, same arithmetic per token, **4.5× the throughput** from coarser
experts — because a `bmm` over 16 groups of 2048 columns fills the hardware
better than one over 64 groups of 512. Against the original loop at 64 experts,
the coarse batched config is 85× faster.

This one is a real trade, not a free win. Fine granularity is the whole claim of
DeepSeekMoE — more, smaller experts are meant to buy quality at equal compute.
Coarsening spends that hypothesis to buy tokens. On a budget run where tokens
are the binding constraint, 4.5× more of them is probably the better side of
the trade; on a run testing the architecture itself, it is not. Worth deciding
deliberately rather than inheriting.

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
the price. If it is launch-overhead-bound, the extra arithmetic is unusable and
the cheapest card with enough memory wins. The two hypotheses predict opposite
orderings. The dispatch fix above should have moved this model into the first
camp, but it was measured on MPS, so `survey` is what establishes that on
CUDA.

`calibrate` exists because the token target sets the cosine schedule, which
must be fixed before the run starts. Size it from a guessed MFU and the run
either stops with its learning rate halfway down the cosine or finishes early
with credit unspent.

### What $30 actually buys

At the **pre-fix** 9.8% MFU, $26 of credit on a 304M model costing 2.36
GFLOP/token. Every row here is now a floor rather than an estimate — the
batched dispatch should raise all of them, by a factor `calibrate` will
measure:

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
