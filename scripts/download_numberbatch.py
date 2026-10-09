#!/usr/bin/env python3
"""
Download ConceptNet Numberbatch vectors for use as a static knowledge channel.

Numberbatch is a set of 300-dimensional term embeddings that combine distributional
word vectors with the ConceptNet knowledge graph via retrofitting. OpenMythos uses
them as a frozen per-token prior that is re-injected at every recurrent iteration.

Only the standard library is used here — numpy, requests and tqdm are not
dependencies of this package and are deliberately not introduced.

LICENSE NOTE: Numberbatch is distributed under CC-BY-SA 4.0, which is not the
MIT license this repository carries. Do not commit the downloaded file or any
table derived from it. Attribution: Robyn Speer, Joshua Chin, and Catherine Havasi
(2017), "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge".

Run:
    python scripts/download_numberbatch.py
    python scripts/download_numberbatch.py --out-dir data/numberbatch
    python scripts/download_numberbatch.py --multilingual
    python scripts/download_numberbatch.py --url <mirror> --force
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

VERSION = "19.08"
BASE = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch"

URL_EN = f"{BASE}/numberbatch-en-{VERSION}.txt.gz"
URL_MULTI = f"{BASE}/numberbatch-{VERSION}.txt.gz"

EXPECTED_DIM = 300
CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _human(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < step or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= step
    return f"{n:.1f}GB"


def download(url: str, dest: str, force: bool = False, verify_fn=None) -> str:
    """
    Stream `url` to `dest`, writing to a temp file and renaming on success.

    The temp-file + os.replace pattern mirrors `save_checkpoint` in
    `training/3b_fine_web_edu.py`: a kill mid-download leaves any previously
    completed file intact rather than a truncated one that later looks valid.
    The `.part` file is removed on any failure (including Ctrl-C), and the
    rename only happens once the byte count matches the server's
    Content-Length (when one was sent) and `verify` has decompressed the whole
    archive. If `dest` already exists it is not re-downloaded, but it is
    still run through `verify`, so a truncated file from an older run is
    rejected rather than silently reused.

    Args:
        url       -- source URL
        dest      -- final path to write
        force     -- re-download even if `dest` already exists
        verify_fn -- archive check to run before the rename and on the
                     skip-if-exists path; defaults to the Numberbatch header
                     check below. download_conceptnet_edges.py passes its own.

    Returns:
        The path written (== dest).
    """
    verify_fn = verify_fn or verify
    if os.path.exists(dest) and not force:
        print(f"Already present, skipping download: {dest} ({_human(os.path.getsize(dest))})")
        print("Pass --force to re-download.")
        verify_fn(dest)
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"

    print(f"Downloading {url}")
    replaced = False
    progress_shown = False
    try:
        try:
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except ValueError:
                    total = 0  # malformed length: fall back to the gzip check alone
                seen = 0
                next_report = 0
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    seen += len(chunk)
                    if seen >= next_report:
                        progress_shown = True
                        if total:
                            pct = 100.0 * seen / total
                            print(
                                f"  {_human(seen)} / {_human(total)} ({pct:.1f}%)",
                                end="\r",
                                flush=True,
                            )
                        else:
                            print(f"  {_human(seen)}", end="\r", flush=True)
                        next_report = seen + 16 * CHUNK
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"Network error fetching {url}: {exc.reason}") from exc
        except (OSError, http.client.HTTPException) as exc:
            # Connection resets, timeouts, IncompleteRead, and local write
            # errors (disk full, permissions) all land here.
            raise SystemExit(
                f"Download of {url} failed: {exc!r}. "
                "Re-run the same command to retry."
            ) from exc

        if progress_shown:
            print()
            progress_shown = False

        # A server or proxy that closes early makes resp.read return b"" rather
        # than raise, so the loop above ends normally on a truncated body.
        if total and seen != total:
            raise SystemExit(
                f"Incomplete download of {url}: received {seen:,} of {total:,} bytes. "
                "Re-run the same command to retry."
            )

        # Catches truncation when no Content-Length was sent (chunked responses).
        verify_fn(tmp)
        os.replace(tmp, dest)
        replaced = True
    except KeyboardInterrupt:
        progress_shown = False
        print(f"\nInterrupted; removing partial download {tmp}", file=sys.stderr)
        raise
    finally:
        if progress_shown:
            print()  # end the \r progress line before any error message
        if not replaced and os.path.exists(tmp):
            os.remove(tmp)

    print(f"Wrote {dest} ({_human(os.path.getsize(dest))})")
    return dest


def check_gzip_magic(path: str, rerun: str) -> None:
    """
    Fail before decompressing anything if `path` is not a gzip file at all.

    The first two bytes catch the common failure: an HTML error page or a
    redirect notice saved under a .gz name. Shared with
    download_conceptnet_edges.py, which passes its own `rerun` hint so the
    operator is told which script to re-run.

    Args:
        path  -- file to inspect
        rerun -- command fragment appended to the failure message
    """
    try:
        with open(path, "rb") as raw:
            magic = raw.read(2)
    except OSError as exc:
        raise SystemExit(f"{path} is not readable: {exc}") from exc
    if magic != b"\x1f\x8b":
        raise SystemExit(f"{path} is not a gzip file (first bytes {magic!r}); {rerun}")

    print(f"Verifying {path} (decompressing the whole archive)...", flush=True)


def drain(fh) -> int:
    """
    Read `fh` to EOF in chunks and return how many bytes came out.

    Reading to the end is the point -- a missing gzip trailer or a CRC
    mismatch raises here rather than thousands of rows into a build.
    """
    total = 0
    while True:
        chunk = fh.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
    return total


def verify(path: str) -> tuple[int, int]:
    """
    Check that `path` is a complete Numberbatch gzip archive.

    Three checks, in order:
      1. the first two bytes are the gzip magic number (catches an HTML error
         page saved under a .gz name);
      2. the first decompressed line is a word2vec-style header
         `<term_count> <dimensions>` with a positive integer count and
         dimensions == EXPECTED_DIM;
      3. the entire decompressed stream is read to EOF in binary chunks, so a
         truncated archive (missing gzip trailer) or a mid-stream corruption
         (bad deflate data or CRC mismatch) is caught here rather than as a
         raw EOFError later in build_concept_table.py.

    The lines after the header are not parsed; term count and vector widths
    of individual rows are not checked.

    Args:
        path -- path to the .txt.gz file

    Returns:
        (term_count, dimensions)
    """
    rerun = "re-run scripts/download_numberbatch.py with --force"
    check_gzip_magic(path, rerun)

    try:
        with gzip.open(path, "rb") as fh:
            line = fh.readline(4096)
            try:
                header = line.decode("utf-8").split()
            except UnicodeDecodeError:
                raise SystemExit(
                    f"Unexpected header in {path!r}: first line is not UTF-8 text "
                    f"({line[:40]!r}); {rerun}"
                ) from None

            if len(header) != 2:
                raise SystemExit(f"Unexpected header in {path!r}: {header!r}; {rerun}")
            try:
                count, dim = int(header[0]), int(header[1])
            except ValueError:
                raise SystemExit(
                    f"Unexpected header in {path!r}: {header!r} is not two integers; {rerun}"
                ) from None
            if count <= 0:
                raise SystemExit(f"Unexpected header in {path!r}: term count {count}; {rerun}")
            if dim != EXPECTED_DIM:
                raise SystemExit(f"Expected {EXPECTED_DIM}-dim vectors, header says {dim}")

            drain(fh)
    except (EOFError, OSError, zlib.error) as exc:
        raise SystemExit(f"{path} is truncated or corrupt ({exc}); {rerun}") from exc

    print(f"Verified: {count:,} terms x {dim} dims")
    return count, dim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dest_for(url: str, out_dir: str, example: str) -> str:
    """
    Destination path for `url`, named from its path component only.

    Naming from the path alone keeps a query string or fragment in a --url
    mirror out of the filename.

    Args:
        url     -- source URL
        out_dir -- directory to write into
        example -- filename quoted in the error when `url` has no path

    Returns:
        The path to write.
    """
    name = os.path.basename(urllib.parse.urlparse(url).path)
    if not name:
        raise SystemExit(
            f"Cannot derive a filename from {url!r}; pass a --url whose path ends "
            f"in a filename, e.g. .../{example}"
        )
    return os.path.join(out_dir, name)


def add_common_args(p: argparse.ArgumentParser, out_dir: str) -> None:
    """The options both download scripts take; only the --out-dir default differs."""
    p.add_argument(
        "--out-dir",
        default=out_dir,
        help="directory to write the archive into",
    )
    p.add_argument("--url", default=None, help="override the source URL entirely")
    p.add_argument(
        "--force", action="store_true", help="re-download even if the file exists"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(p, "data/numberbatch")
    p.add_argument(
        "--multilingual",
        action="store_true",
        help="fetch the full multilingual file (~9.1M terms) instead of English-only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    url = args.url or (URL_MULTI if args.multilingual else URL_EN)
    dest = dest_for(url, args.out_dir, "numberbatch-en-19.08.txt.gz")

    # download() verifies the archive on both the fresh and skip-if-exists paths.
    download(url, dest, force=args.force)

    print()
    print("Next step — align the vectors to the tokenizer vocabulary:")
    print(f"  python scripts/build_concept_table.py --input {dest} --out data/concept_table.pt")


if __name__ == "__main__":
    sys.exit(main())
