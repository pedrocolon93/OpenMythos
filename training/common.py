#!/usr/bin/env python3
"""
The parts of an OpenMythos pretraining run that do not depend on model size.

Split out of `3b_fine_web_edu.py` when `300m_fine_web_edu.py` was added: the
streaming FineWeb-Edu loader, the warmup/cosine schedule and the FSDP-aware
checkpoint pair were identical between them, and a second copy of the
checkpoint logic in particular is the kind of duplication that drifts silently
— the two halves of save/load must agree about the FSDP state-dict context or
a resumed run diverges without erroring.

What stays in each training script is everything a run pins: the variant, the
tokenizer, the batch shape, the token budget, the schedule constants. Those are
deliberately literal per script, so a run's settings can be read off one file.

Scripts import this as a sibling module (`training/` goes on sys.path when a
script in it is run directly), the same pattern `scripts/` already uses to
share the download machinery.
"""

from __future__ import annotations

import math
import os
import time

import torch
from loguru import logger
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import IterableDataset, get_worker_info

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → half-cosine decay to `min_lr`.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to `min_lr` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)

    Args:
        step    -- current global optimizer step (0-indexed)
        warmup  -- number of warmup steps before cosine decay begins
        total   -- step at which the cosine reaches `min_lr`
        max_lr  -- peak learning rate reached at the end of warmup
        min_lr  -- floor learning rate at and after `total` steps

    Returns:
        Scalar learning rate for this step.
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup, since both pick
    the last element of this list.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []

    Returns:
        Sorted list of absolute paths to matching checkpoint files.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- the optimizer whose state should round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config object; saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time; saved for sanity-check
                       on load against a (possibly updated) tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain; older ones
                       are unlinked after a successful write

    Returns:
        None. Writes to disk as a side effect on master rank.
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    """
    Restore model + optimizer from disk, returning the step to resume at.

    Every rank reads the file (`rank0_only=False` on load) so FSDP has access
    to the full state on each rank — the complement to the `rank0_only=True`
    save path. Must mirror save's single-context pattern; splitting the model
    and optimizer loads across two `state_dict_type` blocks has historically
    produced optimizer state bound to the wrong shard shapes.

    `weights_only=False` is required because the checkpoint contains the
    pickled `cfg` dataclass — flip to `weights_only=True` only if you
    separate config out.

    Args:
        model     -- same FSDP-wrapped or raw model used during save
        optimizer -- freshly constructed optimizer to be filled in-place
        path      -- absolute path to a `step_{N:07d}.pt` file produced by
                     `save_checkpoint`
        ddp       -- whether the model is FSDP-wrapped; must match the save run

    Returns:
        The step number the checkpoint was taken at; the caller advances the
        training loop from this value.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Wall-clock budget
# ---------------------------------------------------------------------------


class Budget:
    """
    A wall-clock stop for a run that is paying by the second.

    Rented-GPU runs have a money limit, not a step limit, and the two do not
    translate until throughput is measured. This holds the deadline so the
    loop can stop cleanly — write a checkpoint, exit, release the container —
    instead of being killed by a platform timeout partway through a step with
    the last checkpoint hundreds of steps behind.

    It is a stop, not a schedule. A run that halts here halts with its
    learning rate still partway down the cosine, which costs real loss; the
    deadline is there so a bad throughput estimate costs money instead of
    costing the whole run.

    Args:
        max_hours   -- hours of execution allowed, or None for no limit
        rate_per_hr -- GPU price per hour, used only to report spend
    """

    def __init__(self, max_hours: float | None, rate_per_hr: float = 0.0):
        self.max_hours = max_hours
        self.rate_per_hr = rate_per_hr
        self.t_start = time.perf_counter()

    @property
    def elapsed_hours(self) -> float:
        return (time.perf_counter() - self.t_start) / 3600.0

    @property
    def spent(self) -> float:
        return self.elapsed_hours * self.rate_per_hr

    def exhausted(self) -> bool:
        return self.max_hours is not None and self.elapsed_hours >= self.max_hours

    def summary(self) -> str:
        if self.max_hours is None:
            return f"{self.elapsed_hours:.2f}h elapsed, ${self.spent:.2f} spent"
        return (
            f"{self.elapsed_hours:.2f}h / {self.max_hours:.2f}h elapsed, "
            f"${self.spent:.2f} spent"
        )


def active_params_per_token(model, cfg) -> int:
    """
    Parameters a single token's forward pass actually touches.

    Not the parameter count. Two things make the stored total the wrong
    number for a FLOP estimate here, and they pull in opposite directions:
    the MoE FFN stores `n_experts` experts but routes each token to
    `n_experts_per_tok` of them, and the recurrent block stores one copy of
    its weights but applies them `max_loop_iters` times. For the 300M config
    the first divides the expert bank by 16 and the second multiplies the
    recurrent block by 8, ending at 394M touched per token against 304M
    stored.

    Used for the 6N FLOP estimate and so for MFU, which is the number that
    says whether a rented GPU is being used or merely rented. Expect the
    estimate to be approximate: it counts matmul parameters and ignores
    attention's quadratic term, normalization and the router itself.

    Args:
        model -- an unwrapped OpenMythos (call before FSDP wrapping)
        cfg   -- its MythosConfig

    Returns:
        Parameter equivalents touched per token, forward only.
    """
    routed = shared = recurrent_rest = elsewhere = 0
    for name, p in model.named_parameters():
        if "routed_experts" in name:
            routed += p.numel()
        elif "shared_expert" in name:
            shared += p.numel()
        elif name.startswith("recurrent"):
            recurrent_rest += p.numel()
        else:
            # Embedding included: it is weight-tied to the output head, so a
            # token pays for it once on the way in and once on the way out.
            elsewhere += p.numel()

    per_expert = routed / max(cfg.n_experts, 1)
    per_pass = shared + cfg.n_experts_per_tok * per_expert + recurrent_rest
    return int(elsewhere + cfg.max_loop_iters * per_pass)
