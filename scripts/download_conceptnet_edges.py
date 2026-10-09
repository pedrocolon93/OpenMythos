#!/usr/bin/env python3
"""
Download the ConceptNet assertions dump: the edges Numberbatch only implies.

The vectors fetched by download_numberbatch.py are retrofitted onto this graph,
so relations are present in them only as geometry. This file carries the graph
itself — typed, weighted edges such as IsA, PartOf and UsedFor — which is what a
retrieval policy needs in order to walk from a concept in the text to a related
concept that is not.

The archive is a gzipped 5-column TSV with no header, one assertion per row:

    /a/[/r/IsA/,/c/en/dog/,/c/en/pet/]  /r/IsA  /c/en/dog  /c/en/pet  {...json...}

The JSON field carries the weight, the source dataset and the license. The file
is large (roughly 1.2GB compressed, 34M assertions across all languages), so
expect the download to take a while and the verification pass a few minutes.

Only the standard library is used, and the streaming/verify machinery is shared
with download_numberbatch.py rather than copied.

LICENSE NOTE: ConceptNet is distributed under CC-BY-SA 4.0, which is not the MIT
license this repository carries. Do not commit the downloaded file or any graph
derived from it. Attribution: Robyn Speer, Joshua Chin, and Catherine Havasi
(2017), "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge".

Run:
    python scripts/download_conceptnet_edges.py
    python scripts/download_conceptnet_edges.py --out-dir data/conceptnet
    python scripts/download_conceptnet_edges.py --url <mirror> --force
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_numberbatch import (  # noqa: E402
    add_common_args,
    check_gzip_magic,
    dest_for,
    download,
    drain,
)

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

VERSION = "5.7.0"
URL = f"https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-{VERSION}.csv.gz"

EXPECTED_FIELDS = 5


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(path: str) -> int:
    """
    Check that `path` is a complete ConceptNet assertions archive.

    Same three-stage shape as the Numberbatch check, adapted to a headerless
    TSV:
      1. the first two bytes are the gzip magic number, which catches an HTML
         error page saved under a .gz name;
      2. the first row splits into five tab-separated fields whose edge URI
         starts with "/a/", relation with "/r/" and both endpoints with "/c/",
         so a file of the wrong shape fails here rather than mid-build;
      3. the entire decompressed stream is read to EOF, so truncation (missing
         gzip trailer) or mid-stream corruption is caught now rather than
         thousands of edges into build_concept_graph.py.

    Rows after the first are not parsed, and no count is checked against the
    release notes.

    Args:
        path -- path to the .csv.gz file

    Returns:
        The number of decompressed bytes read.
    """
    rerun = "re-run scripts/download_conceptnet_edges.py with --force"
    check_gzip_magic(path, rerun)

    try:
        with gzip.open(path, "rb") as fh:
            line = fh.readline(8192)
            try:
                fields = line.decode("utf-8").rstrip("\n").split("\t")
            except UnicodeDecodeError:
                raise SystemExit(
                    f"Unexpected first row in {path!r}: not UTF-8 text ({line[:60]!r}); {rerun}"
                ) from None
            if len(fields) != EXPECTED_FIELDS:
                raise SystemExit(
                    f"Unexpected first row in {path!r}: {len(fields)} tab-separated fields, "
                    f"expected {EXPECTED_FIELDS}; {rerun}"
                )
            uri, rel, start, end = fields[:4]
            if not uri.startswith("/a/") or not rel.startswith("/r/"):
                raise SystemExit(
                    f"Unexpected first row in {path!r}: edge {uri!r} relation {rel!r}; {rerun}"
                )
            if not start.startswith("/c/") or not end.startswith("/c/"):
                raise SystemExit(
                    f"Unexpected first row in {path!r}: endpoints {start!r} -> {end!r}; {rerun}"
                )
            total = len(line) + drain(fh)
    except (EOFError, OSError, zlib.error) as exc:
        raise SystemExit(f"{path} is truncated or corrupt ({exc}); {rerun}") from exc

    print(f"Verified: {total / (1 << 30):.2f}GB of assertions, first row {rel} {start} -> {end}")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(p, "data/conceptnet")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    url = args.url or URL
    dest = dest_for(url, args.out_dir, "conceptnet-assertions-5.7.0.csv.gz")

    # download() runs verify on both the fresh and the skip-if-exists path.
    download(url, dest, force=args.force, verify_fn=verify)

    print()
    print("Next step — build the walkable graph against a concept table:")
    print(
        f"  python scripts/build_concept_graph.py --edges {dest} "
        "--table data/concept_table_gpt2.pt --out data/concept_graph_gpt2.pt"
    )


if __name__ == "__main__":
    sys.exit(main())
