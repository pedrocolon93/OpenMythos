#!/usr/bin/env python3
"""
Align ConceptNet Numberbatch vectors to tokenizer spans and save the lookup tables.

Produces two lookup paths that the model merges per position:

  unigram  a (vocab_size, concept_dim) float16 table indexed by token id, for
           terms the tokenizer emits as a single word-initial token
  spans    token-id n-grams of length 2..max-span, for terms whose tokenization
           covers several tokens

The span path is what handles a word broken into subwords, "photos|ynthesis",
as well as a multi-word phrase, "machine learning". Both are just n-grams of
token ids, so one mechanism covers them. A span delivers its vector at the
span's LAST token, the first position where the whole span has been seen;
earlier tokens of the span never receive it, which keeps the channel causal.
Per position the model combines the unigram row with the spans ending there
(one candidate slot per span length), according to cfg.concept_combiner.

Rows for uncovered tokens stay exactly zero. An uncovered position therefore
gets exactly zero from "mean" and "attend", because every candidate is zero and
the projection is bias-free. The runtime does still keep a validity mask: it
derives slot validity from non-zero unigram rows and from span matches, and the
attention combiners use it to exclude empty slots. "cross" can still read the
concepts of earlier positions at a position that has none of its own. The
"mask" saved below only reports coverage; the model does not read it.

Two passes are made over the vector file. The first collects the term list so
every term can be tokenized in batches; the second reads vectors only for the
terms that survived. Both passes apply the same --lang filter.

LICENSE NOTE: Numberbatch is CC-BY-SA 4.0 while this repository is MIT. The
tables produced here are a derivative work of the vectors. Do not commit them.

Run:
    python scripts/build_concept_table.py \\
        --input data/numberbatch/numberbatch-en-19.08.txt.gz \\
        --out data/concept_table.pt
    python scripts/build_concept_table.py --input ... --out ... --max-span 6
    python scripts/build_concept_table.py --input ... --out ... --tokenizer gpt2
    python scripts/build_concept_table.py --input ... --out ... --lang fr
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import unicodedata
import zlib

import torch

# Allow running from a source checkout without installing the package.
try:
    from open_mythos.tokenizer import DEFAULT_MODEL_ID
except ImportError:  # pragma: no cover - convenience path for dev checkouts
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from open_mythos.tokenizer import DEFAULT_MODEL_ID

from transformers import AutoTokenizer

CONCEPT_DIM = 300
NUMBERBATCH_VERSION = "19.08"
NUMBERBATCH_LICENSE = "CC-BY-SA 4.0"
TOKENIZE_BATCH = 20_000


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def strip_lang_prefix(term: str) -> tuple[str | None, str]:
    """
    Split a leading ConceptNet URI prefix off a term when present.

    The multilingual release writes terms as `/c/en/word`; the English-only
    release writes them bare. The language is returned rather than discarded
    because the same surface form exists in several languages (`/c/en/chat`,
    `/c/fr/chat`), and merging them would give English tokens French vectors.
    Callers keep a term only when its language is None or the requested one.

    Returns:
        (lang, term) where lang is None for an unprefixed term.
    """
    if term.startswith("/c/"):
        parts = term.split("/", 3)
        if len(parts) == 4:
            return parts[2], parts[3]
    return None, term


def corrupt_file_exit(path: str) -> SystemExit:
    """Build the error raised when the gzip stream cannot be read to the end."""
    return SystemExit(
        f"{path} is truncated or corrupt; re-run scripts/download_numberbatch.py with --force"
    )


# Errors a damaged .gz can raise mid-read. gzip.BadGzipFile is an OSError.
CORRUPT_FILE_ERRORS = (EOFError, OSError, zlib.error, UnicodeDecodeError)


def is_usable_term(term: str) -> bool:
    """
    Reject terms with no alphabetic content.

    Pure punctuation and bare numerals carry nothing a knowledge graph can add
    over what the token embedding already learns.
    """
    return any(unicodedata.category(ch).startswith("L") for ch in term)


def read_terms(path: str, lang: str) -> tuple[list[str], int]:
    """
    First pass: collect every term in the file, without parsing any vectors.

    Args:
        path -- .txt.gz vector file
        lang -- language code kept from `/c/<lang>/` prefixed terms; unprefixed
                terms are always kept

    Returns:
        (terms, dim) where terms preserves file order.
    """
    terms: list[str] = []
    prefixed = 0
    kept_prefixed = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = fh.readline().split()
            if len(header) != 2 or not header[1].isdigit():
                raise SystemExit(f"Unexpected header in {path!r}: {header!r}")
            dim = int(header[1])
            for line in fh:
                space = line.find(" ")
                if space <= 0:
                    continue
                term_lang, term = strip_lang_prefix(line[:space])
                if term_lang is not None:
                    prefixed += 1
                    if term_lang != lang:
                        continue
                    kept_prefixed += 1
                if is_usable_term(term):
                    terms.append(term)
    except CORRUPT_FILE_ERRORS:
        raise corrupt_file_exit(path) from None

    if prefixed and not kept_prefixed:
        raise SystemExit(
            f"{path} has {prefixed:,} language-prefixed terms but none for "
            f"--lang {lang!r}. Pass the language code the file uses, e.g. --lang en."
        )
    return terms, dim


def normalize_surface(text: str) -> str | None:
    """
    Reduce a decoded token to the form Numberbatch uses for its terms.

    Numberbatch English terms are lowercase with underscores joining phrase
    words. Stripping and lowercasing lets differently cased forms of a token
    (" Photos", " photos") share a vector. It would also make a word-initial
    token and a bare continuation piece with the same spelling resolve to the
    same term. That sharing is NOT wanted, since a continuation like "cat" in
    con|cat|en|ate is not the word cat; build_unigram_index filters those
    pieces out before they get here.

    Args:
        text -- decoded surface string for a single token

    Returns:
        The normalized term, or None if this token should not be looked up.
    """
    s = text.strip().lower()
    if not s or "�" in s:
        return None
    if not is_usable_term(s):
        return None
    return "_".join(s.split())


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def word_initial_probe(tokenizer):
    """
    Build a test for whether a token starts a new word.

    Decoding a token on its own is not enough. Byte-level BPE keeps the space
    (" photos"), but SentencePiece tokenizers such as T5 and DeBERTa-v2 strip
    the leading space from a lone piece, so "▁photo" decodes to "photo",
    exactly like the continuation "synthesis". WordPiece marks continuations
    with "##" instead and never decodes a space either way.

    Every one of these conventions shows up once the token follows another
    word: "a" + "▁photo" decodes to "a photo", while "a" + "synthesis"
    decodes to "asynthesis". The probe therefore decodes the token after an
    anchor word and checks for whitespace at the join.

    Returns:
        A function mapping a token string to True when it is word-initial.
    """
    anchor = tokenizer.tokenize("a")
    anchor_surface = tokenizer.convert_tokens_to_string(anchor) if anchor else ""

    def is_word_initial(token: str) -> bool:
        # The anchor may itself be several pieces (T5 gives "▁", "a").
        if anchor_surface and not anchor_surface[-1:].isspace():
            joined = tokenizer.convert_tokens_to_string(anchor + [token])
            if joined.startswith(anchor_surface):
                return joined[len(anchor_surface) : len(anchor_surface) + 1].isspace()
        # No usable anchor: fall back to the token decoded on its own.
        return tokenizer.convert_tokens_to_string([token])[:1].isspace()

    return is_word_initial


def build_unigram_index(
    tokenizer, vocab_size: int, bare_unigrams: bool = False
) -> dict[str, list[int]]:
    """
    Map each normalized token surface form to every token id producing it.

    Only WORD-INITIAL tokens are indexed: a token that decodes with whitespace
    in front of it when it follows another word (see word_initial_probe). This
    keeps " photos" and drops continuation pieces such as "ynthesis" or the
    "cat" inside con|cat|en|ate, which would otherwise receive the vector of an
    unrelated word. Matching stays causal: the rule looks at the token alone.

    The cost is honest to state: with byte-level BPE a word at the very start
    of a document has no leading space, tokenizes to the bare piece, and gets
    no unigram row. (SentencePiece prepends the word marker at the start of a
    document, so it does not lose that word.) Spans are unaffected, because
    tokenize_terms encodes terms both bare and with a leading space.

    Several ids can still share one surface form (differing casings, or a
    piece with a leading newline), so the value is a list.

    Args:
        tokenizer     -- HuggingFace tokenizer
        vocab_size    -- ids at or above this are not indexed
        bare_unigrams -- also index continuation pieces (the old behaviour)

    Raises:
        SystemExit when the word-initial rule finds no usable token at all,
        which means the tokenizer uses a word-boundary convention the probe
        does not recognize; an empty unigram table would otherwise pass as
        low coverage.
    """
    index: dict[str, list[int]] = {}
    special = set(tokenizer.all_special_ids or [])
    is_word_initial = word_initial_probe(tokenizer)
    continuation_terms = 0

    for tid in range(vocab_size):
        if tid in special:
            continue
        token = tokenizer.convert_ids_to_tokens(tid)
        if token is None:
            continue
        try:
            term = normalize_surface(tokenizer.convert_tokens_to_string([token]))
            if term is None:
                continue
            word_initial = bare_unigrams or is_word_initial(token)
        except Exception:
            continue
        if not word_initial:
            continuation_terms += 1
            continue
        index.setdefault(term, []).append(tid)

    if not index and continuation_terms:
        raise SystemExit(
            f"No word-initial tokens found in {type(tokenizer).__name__} "
            f"({continuation_terms:,} usable tokens were all classed as continuation "
            "pieces), so the unigram table would be empty. This tokenizer's word "
            "boundaries are not recognized; pass --bare-unigrams to index every token."
        )
    return index


def casing_forms(phrase: str) -> list[str]:
    """
    Spellings of a phrase that running text commonly uses.

    Numberbatch stores terms lowercase, but text writes "Machine learning" at
    the start of a sentence and "Neural Network" or "New York" as a name. Each
    casing tokenizes to different ids, so each needs its own span entry.
    str.title() is deliberately avoided: it capitalizes after apostrophes and
    digits ("don't" -> "Don'T").

    Returns:
        Distinct forms in order: lowercase, sentence case, per-word capitals.
    """
    low = phrase.lower()
    forms = [
        low,
        low[:1].upper() + low[1:],
        " ".join(w[:1].upper() + w[1:] for w in low.split(" ")),
    ]
    return list(dict.fromkeys(forms))


def tokenize_terms(
    tokenizer, terms: list[str], max_span: int, vocab_size: int
) -> dict[str, list[tuple[int, ...]]]:
    """
    Tokenize every term and keep the id sequences short enough to match.

    Each term is tokenized in several casings (see casing_forms), and each
    casing twice, bare and with a leading space, because byte-level BPE
    encodes a word differently at the start of a sentence than mid-sentence.
    Every spelling maps to the same vector.

    Sequences of length 1 are skipped: the unigram table already serves them
    with a direct gather, which is cheaper than a search.

    Args:
        tokenizer  -- HuggingFace tokenizer
        terms      -- term strings from the vector file
        max_span   -- longest span to keep
        vocab_size -- ids at or above this are dropped, since the model's
                      embedding does not have rows for them

    Returns:
        term -> list of distinct id tuples, each of length 2..max_span
    """
    out: dict[str, list[tuple[int, ...]]] = {}
    if max_span < 2:
        return out

    for start in range(0, len(terms), TOKENIZE_BATCH):
        chunk_terms = terms[start : start + TOKENIZE_BATCH]

        # Flatten every spelling of every term into one batch, remembering
        # which slice of the batch belongs to which term.
        variants: list[str] = []
        slices: list[tuple[int, int]] = []
        for term in chunk_terms:
            first = len(variants)
            for form in casing_forms(term.replace("_", " ")):
                variants.append(form)
                variants.append(" " + form)
            slices.append((first, len(variants)))
        encoded = tokenizer(variants, add_special_tokens=False)["input_ids"]

        for term, (lo, hi) in zip(chunk_terms, slices):
            seqs = []
            for ids in encoded[lo:hi]:
                if not (2 <= len(ids) <= max_span):
                    continue
                if any(t >= vocab_size for t in ids):
                    continue
                tup = tuple(ids)
                if tup not in seqs:
                    seqs.append(tup)
            if seqs:
                out[term] = seqs

        done = min(start + TOKENIZE_BATCH, len(terms))
        print(f"  tokenized {done:,}/{len(terms):,}", end="\r", flush=True)

    print()
    return out


def fill_vectors(
    path: str,
    lang: str,
    unigram_index: dict[str, list[int]],
    span_seqs: dict[str, list[tuple[int, ...]]],
    table: torch.Tensor,
    mask: torch.Tensor,
    max_span: int,
) -> tuple[
    torch.Tensor, dict[int, tuple[list[tuple[int, ...]], list[int]]], int, int, int
]:
    """
    Second pass: parse vectors only for terms that some path needs.

    Both paths use one conflict rule, first writer wins: a token id or id
    tuple reachable from more than one term keeps the vector of the first such
    term in the file. Later claims are skipped and counted.

    Args:
        path          -- .txt.gz vector file
        lang          -- language filter, identical to the one read_terms used
        unigram_index -- surface term -> token ids
        span_seqs     -- term -> id tuples of length >= 2
        table         -- (vocab_size, dim) float16, modified in place
        mask          -- (vocab_size,) bool, modified in place
        max_span      -- longest span kept

    Returns:
        (span_vectors, per_length_entries, matched_terms, unigram_collisions,
        span_collisions) where per_length_entries maps span length ->
        (id tuples, vector row indices).
    """
    span_vectors: list[torch.Tensor] = []
    per_length: dict[int, tuple[list[tuple[int, ...]], list[int]]] = {
        n: ([], []) for n in range(2, max_span + 1)
    }
    seen_spans: set[tuple[int, ...]] = set()
    matched = 0
    unigram_collisions = 0
    span_collisions = 0

    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            dim = int(fh.readline().split()[1])
            for line in fh:
                space = line.find(" ")
                if space <= 0:
                    continue
                term_lang, term = strip_lang_prefix(line[:space])
                if term_lang is not None and term_lang != lang:
                    continue
                ids = unigram_index.get(term)
                seqs = span_seqs.get(term)
                if not ids and not seqs:
                    continue

                values = [float(v) for v in line[space + 1 :].split()]
                if len(values) != dim:
                    continue
                row = torch.tensor(values, dtype=torch.float16)
                matched += 1

                if ids:
                    for tid in ids:
                        # First writer wins, same as spans below, so the
                        # result never depends on which duplicate came last.
                        if mask[tid]:
                            unigram_collisions += 1
                            continue
                        table[tid] = row
                        mask[tid] = True

                if seqs:
                    row_index = None
                    for tup in seqs:
                        # A tuple can be reachable from more than one term; the
                        # first term to claim it wins, so matching stays a
                        # function of the id sequence alone.
                        if tup in seen_spans:
                            span_collisions += 1
                            continue
                        if row_index is None:
                            row_index = len(span_vectors)
                            span_vectors.append(row)
                        seen_spans.add(tup)
                        grams, rows = per_length[len(tup)]
                        grams.append(tup)
                        rows.append(row_index)
    except CORRUPT_FILE_ERRORS:
        raise corrupt_file_exit(path) from None

    stacked = (
        torch.stack(span_vectors)
        if span_vectors
        else torch.zeros(0, dim, dtype=torch.float16)
    )
    return stacked, per_length, matched, unigram_collisions, span_collisions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", required=True, help="numberbatch .txt.gz from the download script")
    p.add_argument("--out", default="data/concept_table.pt", help="output .pt path")
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace tokenizer id whose vocabulary the tables are aligned to",
    )
    p.add_argument(
        "--max-span",
        type=int,
        default=6,
        help="longest token-id span to index; 1 disables phrase matching entirely",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="override row count; defaults to the tokenizer's vocab_size, which is "
        "what training feeds into MythosConfig.vocab_size",
    )
    p.add_argument(
        "--lang",
        default="en",
        help="language kept from /c/<lang>/ prefixed terms in a multilingual file; "
        "terms prefixed with any other language are skipped, unprefixed terms are kept",
    )
    p.add_argument(
        "--bare-unigrams",
        action="store_true",
        help="also give unigram rows to continuation pieces (tokens that do not start "
        "a word); off by default because a piece like the 'cat' in con|cat|en|ate "
        "would get the vector of an unrelated word",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"No such file: {args.input}. Run download_numberbatch.py first.")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # HuggingFace's vocab_size excludes added/special tokens, and that is the
    # number training assigns to cfg.vocab_size. The tables must match the
    # model's embedding row count exactly or lookups silently misalign.
    vocab_size = args.vocab_size or tokenizer.vocab_size
    print(f"Vocabulary rows: {vocab_size:,}")

    print(f"Pass 1: reading terms from {args.input}")
    terms, dim = read_terms(args.input, args.lang)
    if dim != CONCEPT_DIM:
        raise SystemExit(f"Expected {CONCEPT_DIM}-dim vectors, file says {dim}")
    print(f"  {len(terms):,} usable terms")

    unigram_index = build_unigram_index(tokenizer, vocab_size, args.bare_unigrams)
    which = "all" if args.bare_unigrams else "word-initial"
    print(f"Unigram candidates: {len(unigram_index):,} distinct surface forms ({which} tokens)")

    print(f"Tokenizing terms for spans up to {args.max_span}")
    span_seqs = tokenize_terms(tokenizer, terms, args.max_span, vocab_size)
    print(f"  {len(span_seqs):,} terms tokenize to a matchable span")

    table = torch.zeros(vocab_size, CONCEPT_DIM, dtype=torch.float16)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    print("Pass 2: reading vectors")
    span_vectors, per_length, matched, uni_collisions, span_collisions = fill_vectors(
        args.input, args.lang, unigram_index, span_seqs, table, mask, args.max_span
    )

    spans_payload = {}
    for n, (grams, rows) in sorted(per_length.items()):
        if not grams:
            continue
        spans_payload[str(n)] = {
            "grams": torch.tensor(grams, dtype=torch.int32),
            "rows": torch.tensor(rows, dtype=torch.int32),
        }

    covered = int(mask.sum())
    pct = 100.0 * covered / vocab_size if vocab_size else 0.0
    print(f"Matched {matched:,} terms in the vector file")
    print(f"  unigram: {covered:,} of {vocab_size:,} rows ({pct:.1f}%)")
    for n in sorted(int(k) for k in spans_payload):
        print(f"  span {n}: {spans_payload[str(n)]['grams'].shape[0]:,} sequences")
    print(f"  span vectors: {span_vectors.shape[0]:,}")
    print(
        f"  collisions resolved (first writer kept): {uni_collisions:,} unigram rows, "
        f"{span_collisions:,} span sequences"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "table": table,
            "mask": mask,
            "spans": spans_payload,
            "span_vectors": span_vectors,
            "vocab_size": vocab_size,
            "concept_dim": CONCEPT_DIM,
            "max_span": args.max_span,
            "tokenizer_id": args.tokenizer,
            "lang": args.lang,
            "bare_unigrams": args.bare_unigrams,
            "source": os.path.basename(args.input),
            "version": NUMBERBATCH_VERSION,
            "license": NUMBERBATCH_LICENSE,
        },
        args.out,
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f}MB)")

    print()
    print("Use it with:")
    print(f"  cfg = MythosConfig(..., use_concept_injection=True, concept_max_span={args.max_span})")
    print("  model = OpenMythos(cfg)")
    print(f"  model.load_concept_table({args.out!r})")


if __name__ == "__main__":
    sys.exit(main())
