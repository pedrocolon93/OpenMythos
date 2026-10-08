"""
Jev-style typed decisions on top of OpenMythos.

A decision model reads a context, a question and caller-supplied candidates,
and returns probabilities over those candidates in one forward pass instead of
generating text. Each candidate in the input is followed by a slot token; the
hidden state at that slot is scored by a scalar head. Because attention is
causal, a slot sees the context, the question and its own candidate, so every
candidate is scored against the same evidence.

Decision kinds (named as in the Open-Jev dataset):
    choice -- softmax over the slots: a distribution over the candidates
    score  -- the same softmax, over ordinal rubric levels
    noul   -- a single slot read as a sigmoid: P(yes)

Any reasoning happens inside the recurrent loop, so n_loops is the compute
knob: more loops, more serial depth for the same decision.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.main import MythosConfig, OpenMythos


class DecisionModel(nn.Module):
    """OpenMythos backbone with a scalar head read at candidate slot positions."""

    def __init__(self, cfg: MythosConfig, backbone: Optional[OpenMythos] = None):
        """
        Args:
            cfg      -- MythosConfig for the backbone
            backbone -- an existing (e.g. pretrained) OpenMythos; built fresh if None
        """
        super().__init__()
        self.backbone = backbone if backbone is not None else OpenMythos(cfg)
        self.score = nn.Linear(cfg.dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        slot_mask: torch.Tensor,
        n_loops: Optional[int] = None,
        grad_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids -- (B, T) token ids
            slot_mask -- (B, T) bool, True at each candidate's slot token
            n_loops   -- recurrent depth for this pass
            grad_loops-- backprop through only the last grad_loops loops

        Returns:
            (B, T) slot logits; positions that are not slots hold -inf
        """
        h = self.backbone.hidden_states(input_ids, n_loops, grad_loops=grad_loops)
        logits = self.score(h).squeeze(-1).float()
        return logits.masked_fill(~slot_mask, float("-inf"))


def decision_loss(
    logits: torch.Tensor, target: torch.Tensor, is_noul: torch.Tensor
) -> torch.Tensor:
    """
    Mean loss over a batch that mixes choice/score rows with noul rows.

    Args:
        logits  -- (B, T) slot logits from DecisionModel
        target  -- (B, T) target probability at each slot (0 elsewhere); for
                   choice/score rows it sums to 1, for noul rows it is P(yes)
                   at the single slot
        is_noul -- (B,) bool

    Returns:
        Scalar loss: soft cross-entropy for choice/score, BCE for noul
    """
    slots = torch.isfinite(logits)
    ce = -(target * F.log_softmax(logits, dim=-1).masked_fill(~slots, 0)).sum(-1)
    noul_logit = logits.masked_fill(~slots, 0).sum(-1)
    bce = F.binary_cross_entropy_with_logits(
        noul_logit, target.sum(-1), reduction="none"
    )
    return torch.where(is_noul, bce, ce).mean()


def decision_probs(logits: torch.Tensor, is_noul: torch.Tensor) -> torch.Tensor:
    """
    Turn slot logits into the decision's probabilities.

    Returns:
        (B, T): a distribution over the slots for choice/score rows, P(yes) at
        the slot for noul rows, 0 at non-slot positions
    """
    slots = torch.isfinite(logits)
    choice = F.softmax(logits, dim=-1)
    noul = torch.sigmoid(logits.masked_fill(~slots, 0)) * slots
    return torch.where(is_noul.unsqueeze(-1), noul, choice)
