#!/usr/bin/env python3
"""
Build a walkable ConceptNet graph aligned to an existing concept table.

The concept table maps token ids to vectors, which is enough to inject a
concept the text already contains. It is not enough to reach a concept the text
does NOT contain: for that a position needs a node it can start from, and edges
it can follow. This script produces both.

    text position -> seed node -> neighbours -> their vectors

Nodes are English ConceptNet terms that also have a Numberbatch vector, so
every node a walk lands on can be injected. Edges come from the assertions
dump, kept in both directions: a reverse edge gets its own relation id, so
"dog IsA pet" and "pet IsA-of dog" stay distinguishable to a policy that learns
per-relation preferences.

Fan-out is capped (--max-degree) by edge weight. Hub nodes in ConceptNet have
tens of thousands of edges, nearly all of them weak, and an uncapped adjacency
would be both huge and dominated by noise.

Some edges are held out (--held-out) and removed from the adjacency, so a probe
can ask whether retrieval helps on relations the walk cannot simply look up.
That is a weaker hold-out than it sounds: Numberbatch is retrofitted on this
same graph, so a held-out edge still leaves a trace in the vectors themselves.

LICENSE NOTE: ConceptNet and Numberbatch are CC-BY-SA 4.0 while this repository
is MIT. The graph produced here is a derivative work of both. Do not commit it.

Run:
    python scripts/build_concept_graph.py \\
        --edges data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \\
        --vectors data/numberbatch/numberbatch-en-19.08.txt.gz \\
        --table data/concept_table_gpt2.pt \\
        --out data/concept_graph_gpt2.pt
"""

from __future__ import annotations

import argparse
from array import array
import gzip
import os
import sys
import zlib

import torch

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_concept_table import (  # noqa: E402
    CONCEPT_DIM,
    CORRUPT_FILE_ERRORS,
    is_usable_term,
    strip_lang_prefix,
)

CONCEPTNET_VERSION = "5.7.0"
CONCEPTNET_LICENSE = "CC-BY-SA 4.0"

# Relations that carry no usable semantics for a retrieval walk.
DROP_RELATIONS = frozenset(
    {
        "/r/ExternalURL",  # links out to DBpedia and friends, not a concept edge
        "/r/dbpedia/genre",  # the /r/dbpedia/* family is machine-extracted and noisy
    }
)

# Relations a cloze probe can be written for, i.e. ones that read as a sentence.
PROBE_RELATIONS = (
    "/r/IsA",
    "/r/UsedFor",
    "/r/PartOf",
    "/r/HasProperty",
    "/r/CapableOf",
    "/r/AtLocation",
    "/r/MadeOf",
)

PROBE_SEED = 20_260_922


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def english_term(uri: str) -> str | None:
    """
    Reduce a ConceptNet node URI to the surface term a Numberbatch row uses.

    "/c/en/dog" and "/c/en/dog/n/wn/animal" both name the same node here: the
    sense suffix is dropped, because Numberbatch has one vector per term and
    the concept table's own join key is the bare term. Nodes in other languages
    return None.
    """
    if not uri.startswith("/c/en/"):
        return None
    term = uri[6:]
    cut = term.find("/")
    if cut != -1:
        term = term[:cut]
    if not term or not is_usable_term(term):
        return None
    return term


def edge_weight(meta: str) -> float:
    """
    Pull the weight out of an assertion's JSON field without parsing the JSON.

    json.loads on every one of 34 million rows costs minutes; the field is a
    plain number in a flat position, so a substring scan is enough. Anything
    unparseable falls back to 1.0, the ConceptNet default.
    """
    at = meta.find('"weight":')
    if at == -1:
        return 1.0
    at += 9
    end = at
    while end < len(meta) and meta[end] not in ",}":
        end += 1
    try:
        return float(meta[at:end])
    except ValueError:
        return 1.0


def iter_edges(path: str, min_weight: float):
    """
    Yield (relation, start_term, end_term, weight) for English-to-English edges.

    Args:
        path       -- assertions .csv.gz
        min_weight -- drop edges weaker than this

    Yields:
        Tuples with both endpoints already reduced to bare English terms.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    continue
                rel = fields[1]
                if rel in DROP_RELATIONS or rel.startswith("/r/dbpedia/"):
                    continue
                start = english_term(fields[2])
                if start is None:
                    continue
                end = english_term(fields[3])
                if end is None or end == start:
                    continue
                weight = edge_weight(fields[4])
                if weight < min_weight:
                    continue
                yield rel, start, end, weight
    except CORRUPT_FILE_ERRORS:
        raise SystemExit(
            f"{path} is truncated or corrupt; re-run scripts/download_conceptnet_edges.py --force"
        ) from None


def read_vectors(path: str, wanted: set[str]) -> tuple[list[str], torch.Tensor]:
    """
    Read Numberbatch rows for the terms that appear in the graph.

    Args:
        path   -- numberbatch .txt.gz
        wanted -- terms seen on some kept edge

    Returns:
        (terms in file order, (N, CONCEPT_DIM) float16 vectors)
    """
    terms: list[str] = []
    rows: list[torch.Tensor] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = fh.readline().split()
            if len(header) != 2 or int(header[1]) != CONCEPT_DIM:
                raise SystemExit(f"Unexpected header in {path!r}: {header!r}")
            for line in fh:
                space = line.find(" ")
                if space <= 0:
                    continue
                lang, term = strip_lang_prefix(line[:space])
                if lang is not None and lang != "en":
                    continue
                if term not in wanted:
                    continue
                values = [float(v) for v in line[space + 1 :].split()]
                if len(values) != CONCEPT_DIM:
                    continue
                terms.append(term)
                rows.append(torch.tensor(values, dtype=torch.float16))
    except CORRUPT_FILE_ERRORS:
        raise SystemExit(
            f"{path} is truncated or corrupt; re-run scripts/download_numberbatch.py --force"
        ) from None

    stacked = (
        torch.stack(rows) if rows else torch.zeros(0, CONCEPT_DIM, dtype=torch.float16)
    )
    return terms, stacked


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def cap_and_sort(
    src: torch.Tensor,
    dst: torch.Tensor,
    rel: torch.Tensor,
    weight: torch.Tensor,
    n_nodes: int,
    max_degree: int,
) -> dict[str, torch.Tensor]:
    """
    Turn an edge list into CSR adjacency, keeping the strongest edges per node.

    Sorting by (source, -weight) puts every node's edges together with its best
    first, so capping is a slice per node and the runtime can take the top-m
    neighbours without scoring anything.

    Args:
        src, dst, rel, weight -- parallel edge tensors
        n_nodes               -- node count, fixes the ptr length
        max_degree            -- neighbours kept per node

    Returns:
        {"neigh_ptr", "neigh_idx", "neigh_rel", "neigh_w"}
    """
    # Sort by source, then by descending weight within each source. Two stable
    # passes rather than one packed key: weights are unbounded floats, so
    # folding them into an integer sort key would silently misorder hubs.
    order = torch.argsort(-weight, stable=True)
    src, dst, rel, weight = src[order], dst[order], rel[order], weight[order]
    order = torch.argsort(src.to(torch.int64), stable=True)
    src, dst, rel, weight = src[order], dst[order], rel[order], weight[order]

    counts = torch.bincount(src.to(torch.int64), minlength=n_nodes)
    starts = torch.zeros(n_nodes + 1, dtype=torch.int64)
    starts[1:] = torch.cumsum(counts, 0)
    # Rank of each edge within its source's block, so the cap is a mask.
    rank = torch.arange(src.numel(), dtype=torch.int64) - starts[src.to(torch.int64)]
    keep = rank < max_degree
    src, dst, rel, weight = src[keep], dst[keep], rel[keep], weight[keep]

    counts = torch.bincount(src.to(torch.int64), minlength=n_nodes)
    ptr = torch.zeros(n_nodes + 1, dtype=torch.int64)
    ptr[1:] = torch.cumsum(counts, 0)
    return {
        "neigh_ptr": ptr,
        "neigh_idx": dst.to(torch.int32),
        "neigh_rel": rel.to(torch.int8),
        "neigh_w": weight.to(torch.float16),
    }


def low_rank_keys(vectors: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project node vectors to a narrow key space used for scoring before gathering.

    A walk scores many more candidates than it keeps. Scoring on 32 dimensions
    and gathering the full 300 only for the winners is what keeps per-position,
    per-iteration retrieval affordable, so the projection ships with the graph
    rather than being recomputed per run.

    Principal components of a random subsample, which keeps far more of the
    inner-product structure than a random projection at the same width.

    Returns:
        (keys (N, width) float16, basis (CONCEPT_DIM, width) float32)
    """
    g = torch.Generator().manual_seed(PROBE_SEED)
    n = vectors.shape[0]
    sample = vectors[torch.randperm(n, generator=g)[: min(n, 100_000)]].float()
    sample = sample - sample.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(sample, q=width, niter=4)
    keys = (vectors.float() @ v).to(torch.float16)
    return keys, v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--edges", required=True, help="conceptnet-assertions-*.csv.gz")
    p.add_argument(
        "--vectors",
        default="data/numberbatch/numberbatch-en-19.08.txt.gz",
        help="numberbatch .txt.gz, the source of every node vector",
    )
    p.add_argument(
        "--table",
        default="data/concept_table_gpt2.pt",
        help="concept table built by build_concept_table.py; supplies the seed maps",
    )
    p.add_argument("--out", default="data/concept_graph.pt", help="output .pt path")
    p.add_argument("--max-degree", type=int, default=32, help="neighbours kept per node")
    p.add_argument(
        "--min-weight", type=float, default=1.0, help="drop edges weaker than this"
    )
    p.add_argument(
        "--held-out",
        type=int,
        default=20_000,
        help="probe edges removed from the adjacency (0 keeps every edge)",
    )
    p.add_argument("--key-dim", type=int, default=32, help="width of the scoring keys")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Pass 1: scanning {args.edges} for English edges")
    wanted: set[str] = set()
    n_seen = 0
    for _, start, end, _ in iter_edges(args.edges, args.min_weight):
        wanted.add(start)
        wanted.add(end)
        n_seen += 1
    print(f"  {n_seen:,} English edges over {len(wanted):,} distinct terms")
    if not n_seen:
        raise SystemExit(f"No usable English edges in {args.edges!r}")

    print(f"Pass 2: reading vectors for those terms from {args.vectors}")
    node_terms, node_vectors = read_vectors(args.vectors, wanted)
    node_id = {t: i for i, t in enumerate(node_terms)}
    print(f"  {len(node_terms):,} of {len(wanted):,} terms have a vector")
    if not node_terms:
        raise SystemExit("No graph node has a vector; are --edges and --vectors both English?")

    print("Pass 3: building the edge list")
    rel_id: dict[str, int] = {}
    # array, not list: several million Python ints per column would cost an
    # order of magnitude more memory for the same numbers.
    src, dst = array("i"), array("i")
    rel, weight = array("h"), array("f")
    for relation, start, end, w in iter_edges(args.edges, args.min_weight):
        a, b = node_id.get(start), node_id.get(end)
        if a is None or b is None:
            continue
        r = rel_id.setdefault(relation, len(rel_id))
        # Both directions, the reverse carrying its own relation id so a policy
        # can tell "dog IsA pet" from "pet has-instance dog".
        src.append(a)
        dst.append(b)
        rel.append(r)
        weight.append(w)
        src.append(b)
        dst.append(a)
        rel.append(r + 64)
        weight.append(w)
    if len(rel_id) > 64:
        raise SystemExit(
            f"{len(rel_id)} relations exceeds the 64 that fit alongside their inverses in int8"
        )
    rel_names = [""] * len(rel_id)
    for name, i in rel_id.items():
        rel_names[i] = name

    src_t = torch.frombuffer(src, dtype=torch.int32).clone()
    dst_t = torch.frombuffer(dst, dtype=torch.int32).clone()
    rel_t = torch.frombuffer(rel, dtype=torch.int16).clone()
    w_t = torch.frombuffer(weight, dtype=torch.float32).clone()
    del src, dst, rel, weight
    print(f"  {src_t.numel():,} directed edges, {len(rel_id)} relations")

    held_out = torch.zeros(0, 3, dtype=torch.int32)
    if args.held_out:
        probe_ids = {rel_id[r] for r in PROBE_RELATIONS if r in rel_id}
        # Forward direction only: a probe sentence reads "a dog is a kind of pet",
        # never the inverse, and removing one direction would leave the other.
        eligible = torch.isin(rel_t, torch.tensor(sorted(probe_ids), dtype=torch.int16))
        idx = eligible.nonzero(as_tuple=True)[0]
        g = torch.Generator().manual_seed(PROBE_SEED)
        take = idx[torch.randperm(idx.numel(), generator=g)[: args.held_out]]
        # Drop both directions of every held-out pair, or the walk reaches the
        # answer backwards and the probe measures nothing.
        pairs = torch.stack([src_t[take], dst_t[take]], dim=1)
        held_out = torch.cat([pairs, rel_t[take].to(torch.int32).unsqueeze(1)], dim=1)
        key = src_t.to(torch.int64) * len(node_terms) + dst_t.to(torch.int64)
        drop_keys = torch.cat(
            [
                pairs[:, 0].to(torch.int64) * len(node_terms) + pairs[:, 1].to(torch.int64),
                pairs[:, 1].to(torch.int64) * len(node_terms) + pairs[:, 0].to(torch.int64),
            ]
        )
        keep = ~torch.isin(key, drop_keys)
        src_t, dst_t, rel_t, w_t = src_t[keep], dst_t[keep], rel_t[keep], w_t[keep]
        print(f"  held out {held_out.shape[0]:,} probe edges ({int((~keep).sum()):,} directed)")

    print(f"Capping fan-out at {args.max_degree} neighbours per node")
    csr = cap_and_sort(src_t, dst_t, rel_t, w_t, len(node_terms), args.max_degree)
    degree = csr["neigh_ptr"][1:] - csr["neigh_ptr"][:-1]
    print(
        f"  {csr['neigh_idx'].numel():,} edges kept; degree mean {degree.float().mean():.1f}, "
        f"median {int(degree.median())}, max {int(degree.max())}, "
        f"isolated {int((degree == 0).sum()):,}"
    )

    print(f"Projecting {args.key_dim}-d scoring keys")
    node_keys, key_basis = low_rank_keys(node_vectors, args.key_dim)

    print(f"Mapping the concept table in {args.table} onto nodes")
    payload = torch.load(args.table, map_location="cpu", weights_only=False)
    for key in ("span_terms", "unigram_terms"):
        if key not in payload:
            raise SystemExit(
                f"{args.table} has no {key!r}: rebuild it with the current "
                "scripts/build_concept_table.py, which records which term each row came from."
            )
    token_node = torch.full((payload["vocab_size"],), -1, dtype=torch.int32)
    for tid, term in enumerate(payload["unigram_terms"]):
        if term:
            token_node[tid] = node_id.get(term, -1)
    span_node = torch.full((len(payload["span_terms"]),), -1, dtype=torch.int32)
    for row, term in enumerate(payload["span_terms"]):
        span_node[row] = node_id.get(term, -1)
    print(
        f"  {int((token_node >= 0).sum()):,} of {int((payload['mask']).sum()):,} covered token ids "
        f"reach a node; {int((span_node >= 0).sum()):,} of {span_node.numel():,} span rows do"
    )

    out = {
        "node_terms": node_terms,
        "node_vectors": node_vectors,
        "node_keys": node_keys,
        "key_basis": key_basis,
        **csr,
        "token_node": token_node,
        "span_node": span_node,
        "rel_names": rel_names,
        "held_out_edges": held_out,
        "max_degree": args.max_degree,
        "min_weight": args.min_weight,
        "concept_dim": CONCEPT_DIM,
        "tokenizer_id": payload.get("tokenizer_id"),
        "vocab_size": payload["vocab_size"],
        "edges_source": os.path.basename(args.edges),
        "version": CONCEPTNET_VERSION,
        "license": CONCEPTNET_LICENSE,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / (1 << 20):.1f}MB)")

    # Eyeball the graph once rather than trusting it.
    print()
    print("Sample neighbourhoods:")
    for term in ("photosynthesis", "neural_network", "dog", "kettle"):
        i = node_id.get(term)
        if i is None:
            print(f"  {term}: not a node")
            continue
        lo, hi = int(csr["neigh_ptr"][i]), int(csr["neigh_ptr"][i + 1])
        shown = []
        for j in range(lo, min(hi, lo + 6)):
            r = int(csr["neigh_rel"][j])
            name = rel_names[r] if r < 64 else rel_names[r - 64] + "-of"
            shown.append(f"{name.removeprefix('/r/')} {node_terms[int(csr['neigh_idx'][j])]}")
        print(f"  {term} ({hi - lo} edges): " + ", ".join(shown))


if __name__ == "__main__":
    sys.exit(main())
