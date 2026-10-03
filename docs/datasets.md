# Recommended Training Datasets

| Dataset | HuggingFace | Tokens | License | Use |
|---|---|---|---|---|
| FineWeb-Edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T | Apache 2.0 | Primary pretraining |
| OpenHermes 2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~1M samples | Apache 2.0 | Instruction tuning (~5% mix) |
| OpenWebMath | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | 14.7B | ODC-By | Math/reasoning boost |

---

## Primary Pretraining

### FineWeb-Edu
- **HuggingFace:** `HuggingFaceFW/fineweb-edu`
- **Size:** 1.3T tokens
- **License:** Apache 2.0
- **Why:** Web text filtered for educational quality. Outperforms The Pile, C4, and RefinedWeb on downstream benchmarks. Already deduplicated and cleaned.
- **Start with:** `sample-10BT` to validate your pipeline, then `sample-100BT` or the full corpus for a serious run.

## Supplementary

### OpenHermes 2.5
- **HuggingFace:** `teknium/OpenHermes-2.5`
- **Size:** ~1M instruction samples
- **License:** Apache 2.0
- **Why:** High-quality instruction-following data. Mix in ~5% by token count on top of FineWeb-Edu to improve instruction following without degrading general capability.

### OpenWebMath
- **HuggingFace:** `open-web-math/open-web-math`
- **Size:** ~14.7B tokens
- **License:** ODC-By
- **Why:** Math-focused web text. Add if you want stronger quantitative and symbolic reasoning. Particularly useful for the 10B+ variants where reasoning depth matters.

## Token Budget Recommendations

| Variant | Chinchilla-optimal | Recommended (looped) |
|---|---|---|
| 1B | ~20B tokens | ~10–15B tokens |
| 3B | ~60B tokens | ~30–40B tokens |
| 10B | ~200B tokens | ~100–150B tokens |
| 50B+ | ~1T+ tokens | ~500B+ tokens |

The looped architecture is more sample-efficient than a standard transformer — same validation loss is reachable with fewer tokens due to faster convergence. The "recommended (looped)" column reflects this and is based on the Tiny Shakespeare result where OpenMythos reached equivalent loss ~2.5× faster than nanoGPT.

---


## ConceptNet Numberbatch (optional static knowledge channel)

Not a training corpus. Numberbatch supplies 300-dimensional term vectors that
OpenMythos can fuse into the network at configurable sites: the token
embedding (`embed`), the encoded input `e` that the recurrent block re-injects
at every loop iteration (the default), and the recurrent attention input
(`attn`). Each enabled site has its own zero-init gate. Disabled by default.

| Resource | Source | Terms | License |
|---|---|---|---|
| Numberbatch 19.08 (English) | [conceptnet-numberbatch](https://github.com/commonsense/conceptnet-numberbatch) | ~516k | CC-BY-SA 4.0 |

```bash
python scripts/download_numberbatch.py
python scripts/build_concept_table.py \
    --input data/numberbatch/numberbatch-en-19.08.txt.gz \
    --out data/concept_table.pt
```

**Download.** The file is streamed to a `.part` file and renamed into place
only after two checks pass: the number of bytes received matches the server's
`Content-Length` (when the server sends one), and the whole archive
decompresses to the end behind a `<term_count> 300` header. A failed or
interrupted download removes the `.part` file. If the file already exists it
is not downloaded again, but it is still verified, which costs one full
decompression pass; a truncated or corrupt file stops with a message to re-run
with `--force`. There is no checksum and no resume. `--multilingual` fetches
the multilingual file instead.

**Building the table.** `build_concept_table.py` aligns the vectors to one
tokenizer and writes two lookup paths.

| Flag | Default | Effect |
|---|---|---|
| `--tokenizer` | `openai/gpt-oss-20b` | Tokenizer the table is aligned to. Must be the one the model trains with. |
| `--vocab-size` | the tokenizer's `vocab_size` | Row count of the unigram table. Must equal `cfg.vocab_size`. |
| `--max-span` | `6` | Longest token-id span indexed. `1` disables the span path. |
| `--lang` | `en` | For the multilingual file, whose terms look like `/c/fr/chat`: keep only terms prefixed with this language. Other prefixes are skipped; unprefixed terms (the English-only file) are always kept. The script exits if the file has prefixed terms but none for `--lang`. |
| `--bare-unigrams` | off | Also give unigram rows to continuation pieces (see below). |

- **Unigram path.** One row per token id, for terms the tokenizer emits as a
  single *word-initial* token, meaning the token decodes with leading
  whitespace. Surfaces are stripped and lowercased, so ` Photos` and ` photos`
  get the same vector. Continuation pieces get no row by default, because the
  `cat` inside `con|cat|en|ate` would otherwise receive the vector of the
  unrelated word. The cost: a word at the very start of a document has no
  leading space and gets no unigram row. `--bare-unigrams` restores rows for
  continuation pieces.
- **Span path.** Every term is tokenized in three casings (lowercase, sentence
  case such as `Machine learning`, and per-word capitals such as
  `Machine Learning`), each both bare and with a leading space. Id sequences
  of length 2 to `--max-span` are kept.
- **Conflicts.** When one token id or id sequence is reachable from several
  terms, the first term in the file wins, on both paths. The script prints how
  many collisions it resolved.

Then enable it on the model:

```python
from open_mythos import MythosTokenizer, OpenMythos, mythos_3b

cfg = mythos_3b()
cfg.vocab_size = MythosTokenizer().vocab_size  # must match the table's rows
cfg.use_concept_injection = True
cfg.concept_max_span = 6                     # match the build's --max-span
cfg.concept_sites = ("embed", "e", "attn")   # any non-empty subset
cfg.concept_combiner = "cross"               # "mean" | "attend" | "cross"
model = OpenMythos(cfg)
summary = model.load_concept_table(
    "data/concept_table.pt", tokenizer_id="openai/gpt-oss-20b"
)
```

`load_concept_table(source, tokenizer_id=None)` raises `RuntimeError` if
`tokenizer_id` is given and differs from the tokenizer the table was built
for, if the table's `vocab_size` differs from the model's, or if the table
shape does not match the config. Span lengths above `concept_max_span` are
skipped with a warning and listed in `summary["dropped_span_lengths"]`.
Calling the model with the channel enabled but no table loaded raises
`RuntimeError`.

**Sites.** `embed` adds the delta to the token embedding before the Prelude,
`e` adds it to the frozen encoding that the recurrent block re-injects every
iteration, and `attn` adds it to the recurrent attention input on each
iteration. Each site has its own zero-init gate, so they can be enabled
together and ablated separately.

**Combiners.** A position can have several concept vectors available: its own
unigram entry plus every span ending there. `mean` applies a learned weight
per slot, then divides by the number of valid slots (not by the sum of the
weights). `attend` lets a query from the site's hidden state pick among them.
`cross` runs causal cross-attention over the whole sequence's concept memory,
so a token can read a concept that completed earlier.

**Causality.** A span delivers its vector at its *last* token, the first
position where the span is fully observed. Delivering it earlier would hand
the model the identity of its own next-token target. So when the GPT-2
tokenizer splits ` photosynthesis` into ` photos|ynthesis`, the first fragment
receives no span vector, the second receives the word vector, and every later
token can reach it through the `cross` memory.

**Loading and checkpoints.** Every lookup structure is a non-persistent
buffer, so the table never enters `state_dict`: checkpoints stay small and do
not depend on which table is loaded. It must therefore be loaded after
building the model on every run and every distributed rank. Every site's gate
starts at zero, so an untrained model is numerically identical to the baseline
until training moves them. Enabling the channel does add trainable `concept.*`
parameters, so resuming a checkpoint saved without it needs
`model.load_state_dict(..., strict=False)` (the only missing keys should be
`concept.*`) plus a fresh optimizer state.

**Coverage.** Coverage is partial by design. A term the tokenizer emits as one
word-initial token matches through the unigram table. A term split across
several tokens matches through the span table at its last fragment only;
earlier fragments get no span vector, although a word-initial fragment that is
itself a word, like ` photos`, still gets its own unrelated unigram row.
Uncovered unigram rows are all zeros and the projection is bias-free, so under
`mean` and `attend` a position with no candidates gets exactly zero. Under
`cross` an uncovered position can still read concepts that completed earlier
in the sequence, so its delta is not zero.

**Licensing:** Numberbatch is CC-BY-SA 4.0 while this repository is MIT. Neither
the downloaded vectors nor the derived table may be committed; both are covered
by `.gitignore`. Attribution: Robyn Speer, Joshua Chin and Catherine Havasi
(2017), "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge".

## ConceptNet graph (optional retrieval of concepts the text lacks)

The table above can only inject concepts the text already contains. The graph
adds the other direction: from a concept a position has, walk out to related
concepts it does not have, and let the model attend over those too. Numberbatch
was retrofitted on this graph, so the relations are implicit in the vectors,
but only the assertions dump carries them explicitly.

| Resource | Source | Size | License |
|---|---|---|---|
| ConceptNet 5.7 assertions | [conceptnet5 downloads](https://github.com/commonsense/conceptnet5/wiki/Downloads) | ~500MB gzipped | CC-BY-SA 4.0 |

```bash
python scripts/download_conceptnet_edges.py
python scripts/build_concept_graph.py \
    --edges data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --table data/concept_table_gpt2.pt \
    --out data/concept_graph_gpt2.pt
```

**Download.** Same machinery as the vectors: streamed to a `.part` file,
byte count checked against `Content-Length`, whole archive decompressed, and
the first row checked for five tab-separated fields with `/a/`, `/r/` and
`/c/` URIs. No checksum, no resume.

**Building the graph.** The builder needs a table built by the current
`build_concept_table.py`, which records the term behind each row (`span_terms`,
`unigram_terms`). Without those a row is a vector and nothing else, and a walk
has nowhere to start. Rebuild an older table before running this.

| Flag | Default | Effect |
|---|---|---|
| `--max-degree` | `32` | Neighbours kept per node, strongest edges first. ConceptNet hubs have tens of thousands of edges, almost all weak. |
| `--min-weight` | `1.0` | Drop edges below this weight. |
| `--held-out` | `20000` | Probe edges removed from the adjacency entirely. |
| `--key-dim` | `32` | Width of the per-node scoring keys. |

Nodes are English terms that appear in the assertions *and* have a Numberbatch
vector, so anything a walk reaches can also be injected. Edges are stored in
both directions, the reverse carrying its own relation id, and each node's
neighbours are sorted strongest-first so taking the top few needs no scoring.
The output also carries a `--key-dim` projection of every node vector: a walk
scores many more candidates than it keeps, and scoring narrow before gathering
300 dimensions is what keeps per-position retrieval affordable.

**Using it.**

```python
cfg = MythosConfig(
    ..., use_concept_injection=True, concept_max_span=6,
    concept_combiner="attend",   # the walk needs attend; it scores retrieved slots
    concept_walk="fixed",        # "none" (default) or "fixed"
    concept_walk_k=4,            # retrieved concepts kept per position
    concept_walk_fanout=4,       # neighbours considered per seed concept
)
model = OpenMythos(cfg)
model.load_concept_table("data/concept_table_gpt2.pt")
model.load_concept_graph("data/concept_graph_gpt2.pt")   # table first: it is checked against it
```

Retrieved vectors join the position's own candidates in one softmax, so the two
compete for the same attention, and the gate is shared with the site. The
architecture is identical with `concept_walk="none"`, which is what makes that
setting the control to judge a walking run against.

**Causality.** A walk starts only from nodes a position reached causally, so it
inherits the table's causality: no retrieved concept can depend on a later
token. `tests/test_main.py::TestConceptWalk::test_walk_is_causal` asserts the
concept path is bit-identical over a shared prefix.

**Licensing:** the ConceptNet assertions are CC-BY-SA 4.0, as is any graph
derived from them, and `.gitignore` covers `data/`. Attribution as above.
