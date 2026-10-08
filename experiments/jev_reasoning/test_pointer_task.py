"""Checks that the generated decisions are right and need exactly `hops` lookups."""

import numpy as np

from pointer_task import BOS, SLOT, TaskConfig, VAL0, VAR0, Q, make_batch, make_example


def resolve(toks):
    """Follow the asked chain by brute force; return (value, lookups)."""
    q = toks.index(Q)
    table = {}
    for i in range(1, q, 4):
        table[toks[i]] = toks[i + 2]
    cur, n = toks[q + 1], 0
    while cur < VAL0:
        cur, n = table[cur], n + 1
    return cur, n


def test_answer_and_depth():
    rng, cfg = np.random.default_rng(0), TaskConfig()
    for hops in range(1, 17):
        for noul in (False, True):
            for _ in range(50):
                toks, slots, target = make_example(rng, hops, cfg, noul)
                value, n = resolve(toks)
                assert n == hops
                assert toks[0] == BOS and all(toks[s] == SLOT for s in slots)
                for s, y in zip(slots, target):
                    assert y == float(toks[s - 1] == value)
                if not noul:
                    assert sum(target) == 1 and len(slots) == cfg.n_chains


def test_batch_shapes():
    rng, cfg = np.random.default_rng(1), TaskConfig()
    b = make_batch(rng, [1, 5, 12], cfg, [False, True, False])
    assert b["slot_mask"].sum(1).tolist() == [4, 1, 4]
    assert b["input_ids"].shape[1] <= cfg.max_len + 4 * 4
