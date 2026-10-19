"""Shared setup for the diagnostics: import paths, a small model, a plain baseline."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # pointer_task, train
sys.path.insert(0, str(HERE.parents[2]))  # this checkout's open_mythos

import torch
import torch.nn as nn

import train as T
from pointer_task import VOCAB_SIZE, TaskConfig


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def small_config(task: TaskConfig, dim: int = 128):
    """The experiment's OpenMythos config at a size that trains in minutes."""

    class Args:
        arm = "fixed"

    Args.dim = dim
    return T.model_config(Args, task)


class Vanilla(nn.Module):
    """
    Plain pre-norm causal transformer with learned positions: the control for
    "is it the task or is it OpenMythos?". out="slots" returns slot logits like
    DecisionModel; out="lm" returns next-token logits.
    """

    def __init__(self, dim: int = 128, layers: int = 4, out: str = "slots"):
        super().__init__()
        self.out = out
        self.emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(512, dim)
        layer = nn.TransformerEncoderLayer(
            dim, 4, 4 * dim, 0.0, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1 if out == "slots" else VOCAB_SIZE)

    def forward(self, ids, slot_mask=None, n_loops=None, grad_loops=None):
        L = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device))
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=ids.device)
        x = self.norm(self.enc(x, mask=causal, is_causal=True))
        if self.out == "lm":
            return self.head(x)
        return self.head(x).squeeze(-1).float().masked_fill(~slot_mask, float("-inf"))
