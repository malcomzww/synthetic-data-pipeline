# synthetic-data-pipeline

A synthetic instruction-data pipeline whose deliverable is the **hygiene
layer**, not the generation: MinHash/LSH deduplication, 13-gram
decontamination against a held-out eval set, regex PII scrubbing, and
script-level language ID — with provenance carried inside every record.
The generation step is **templated recombination, not model output**; no
language model is called anywhere in this repository. That is deliberate, and
it is what makes the numbers below measurable: the duplicates, the
contamination and the PII are all **planted at known rates**, so every
detector is scored against exact ground truth rather than against another
detector's opinion.

**This repo answers one question:**

> What fraction of a generated set is contaminated against the eval set before
> decontamination?

## The headline

Of 900 records reaching the decontamination stage with **108 contaminated
spans planted (12%)**, a 13-gram overlap detector flagged **63 — a recall of
58.3%, and a miss rate of 41.7%.**

The miss rate is the useful half, and it has exactly one cause:

| planted span | records | caught | missed |
|---|---|---|---|
| ≥ 13 tokens | 63 | 63 | 0 |
| < 13 tokens | 45 | 0 | 45 |

**Detectability is decided entirely by span length, with no exceptions in
either direction.** Every span of at least *n* tokens was found; every shorter
span was missed. That is the detector's definition made visible rather than a
surprising empirical result — an n-gram matcher cannot see a fragment shorter
than *n* — but watching it hold *exactly* is what tells you the implementation
does what it claims.

The consequence is the part worth carrying:

> **A clean decontamination report means "no leaked span of 13+ tokens", not
> "no leakage".** Against shorter fragments this method is not weak, it is
> blind. Reporting the pass rate without that qualifier is how a contaminated
> eval survives review.

Sweeping *n* shows the same mechanism from the other side:

| n | recall | false positives |
|---|---|---|
| 5 | 100.0% | 0 |
| 8 | 91.7% | 0 |
| **13** | **58.3%** | 0 |
| 20 | 30.6% | 0 |

Lowering *n* raises recall here — so why keep 13? Because the false-positive
column is misleadingly clean. This eval set is 24 hand-authored items whose
phrasing is nearly unique; against a real benchmark sharing ordinary English
with the training corpus, a 5-gram matcher flags large amounts of clean text.
**The cost of small *n* is invisible in this fixture**, which is exactly why
13 is taken on convention rather than tuned to whatever maximises recall on
900 synthetic records.

Full numbers, with assertions: [`results/contamination.md`](results/contamination.md).

## Honesty about what was measured

The contamination was **planted at a rate chosen by the experiment**, not
discovered in the wild. Nothing here is evidence that any real dataset is
contaminated. What is measured is **detector recall against ground truth that
is known exactly because it was constructed** — which is a real measurement,
and the only kind available without an LLM budget.

Every record's provenance says `source="templated"`, `model=None`, and a test
asserts it.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q                        # 155 passed
uv run python scripts/generate_results.py
```

The results script **asserts every claim it prints** and exits non-zero if one
breaks. Output is byte-identical across runs and processes for a fixed seed,
so CI's `git diff --exit-code results/` drift gate holds.

## The survival funnel

960 records in, 371 out (38.6% survive), with all three hazards planted at once:

| stage | entered | dropped | survived | why |
|---|---|---|---|---|
| filter | 960 | 3 | 957 | `too_short` 3 |
| dedup | 957 | 521 | 436 | `near_duplicate` 521 |
| decontaminate | 436 | 65 | 371 | `ngram_overlap` 65 |

The funnel is **recomputed from the surviving records' own provenance**, not
from counters incremented as the pipeline ran. A counter can drift from the
data it describes; a value derived from the artifacts cannot. Dropped records
are kept rather than deleted, because the interesting question is not "how
many survived" but "what did we throw away, and was that right?"

See [`docs/lineage.md`](docs/lineage.md) for the provenance policy and its
limits.

## The other detectors

**Near-duplicates (MinHash + LSH, Jaccard 0.7):** 60 planted pairs, **95%
recall**. Reported precision is 1.8% and that figure is an artifact worth
explaining rather than a result. Run the same detector over the base corpus
with *nothing planted*: it finds **3,044 pairs**, against 3,123 after
planting. The template generator emits records differing by a single noun
phrase, and at Jaccard 0.7 those genuinely *are* near-duplicates — the
detector is right to flag them; the scoring function only calls them false
positives because they were not *planted*. **Recall is trustworthy here,
precision is not**, and that is a limitation of the fixture rather than a
finding about MinHash.

**PII scrubbing:** 100% recall and 100% precision against 72 planted
identifiers across six shapes (email, phone, SSN, card, IPv4, IBAN). This
measures whether the regexes fire on the shapes they were written for; it is
**not** evidence a corpus is PII-free. Specimens use documentation-reserved
values only. PII is *scrubbed*, not dropped — removing every record mentioning
an email address would bias the corpus away from anything about contacting
people.

**Script coverage:** 900 base records across 7 scripts (Latin, Cyrillic,
Greek, Devanagari, Arabic, Han, Hangul), with 18 code-switched records
detected. This is **script detection, not language identification** — it
cannot separate Spanish from Portuguese, because they share a script.

## Why this exists

Three things are easy to get wrong in a data pipeline, and all three produce
numbers that look fine:

1. **Scoring a detector on a corpus whose ground truth you do not control.**
   Without planted duplicates you can compare detectors to each other and
   never learn what either one misses.
2. **Building a fixture that guarantees the answer.** The first version of
   this repo planted every contamination span at ≥13 tokens and measured 100%
   recall — a number that described the fixture, not the detector.
3. **Letting one measurement contaminate another.** The first version planted
   duplicates, contamination and PII into a single corpus; the splices pulled
   planted duplicate pairs from Jaccard 0.85 down to 0.32, and MinHash recall
   read 0%. The detector was fine. The corpus had stopped containing
   duplicates.

Each of those was a real bug in this repository, found by an assertion in the
results script rather than by inspection, and each is now covered by a test.

## Limitations

- **Nothing here is contamination found in the wild.** The rate was chosen,
  planted, then recovered. The transferable result is the detector's recall
  and miss rate, not the rate.
- **The corpus is templated**, bounded by a cross product of 10 frames × 10
  domains × 18 objects. MinHash recall on a genuinely model-generated corpus
  — more diverse in some ways, far more repetitive in others — is not
  established here.
- **Word n-grams are blind on unsegmented scripts.** CJK records pass through
  the pipeline, but the 13-gram detector cannot see leakage in them at all.
  This repo does not establish a decontamination method that works for Chinese
  or Japanese, and the multilingual slice is too small to quantify the gap.
- **Near-duplicate precision is not established**, for the fixture reason
  above. A trustworthy precision figure needs a corpus whose *complete*
  duplicate set is known, not one whose planted subset is.
- **PII recall is measured against regex-shaped planted identifiers**, close
  to the best case for a regex scrubber. Unformatted PII — a name, an address
  in prose, a date of birth in words — is neither measured nor caught.
- **The eval set is 24 hand-authored items**, not a real benchmark. Its n-gram
  profile is not MMLU's, and the false-positive rate against a real benchmark
  would differ.
- **No semantic deduplication.** Records meaning the same thing in different
  words survive every stage here.

## Layout

```
src/synthetic_data_pipeline/
  lineage.py         Record, provenance events, survival funnel
  generate.py        templated generator + planted near-duplicates
  eval_set.py        the held-out eval set
  dedup.py           exact hashing + MinHash/LSH
  decontaminate.py   n-gram overlap + planted contamination
  filters.py         length, script ID, PII scrubbing
scripts/generate_results.py    asserts every claim, writes results/
docs/lineage.md                the provenance policy
```

## Concepts covered

See [`docs/inventory-coverage.md`](docs/inventory-coverage.md).

## License

MIT
