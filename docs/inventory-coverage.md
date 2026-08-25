# Inventory coverage

Anchored to `Bucket_Concept_Inventory.md`, bucket B1 (LLM & post-training).

Marked honestly: **built** means implemented and measured here, **partial**
means implemented in a narrower form than the inventory line implies, and
**not covered** means the line is named but this repo does not establish it.

| concept | status | where |
|---|---|---|
| 1C synthetic data generation: self-instruct, evol-instruct, persona, Magpie-style | **partial** | `generate.py` — templated recombination stands in for the LLM loop. No model is called. The `Generator` protocol is the swap seam. |
| 1C synthetic risks: model collapse, distribution narrowing, style leakage, benchmark contamination | **partial** | Benchmark contamination is measured (`decontaminate.py`). Distribution narrowing is visible in the near-duplicate density of the templated corpus. Model collapse and style leakage need a real generation loop and are **not** established. |
| 1C deduplication: exact, MinHash/LSH, semantic | **partial** | `dedup.py` — exact and MinHash/LSH built and scored. **Semantic dedup is not implemented**; records meaning the same thing in different words survive. |
| 1C decontamination via n-gram overlap | **built** | `decontaminate.py` — the repo's headline measurement, with the span-length result that explains its miss rate. |
| 1C filtering: quality classifier, perplexity, toxicity, language ID, PII/PHI scrubbing | **partial** | `filters.py` — length, script-level language ID and regex PII scrubbing built and scored. **No quality classifier, no perplexity filter, no toxicity model**: each needs a model download this repo deliberately avoids. |
| 1C multilingual and low-resource: script coverage, transliteration, code-switching | **partial** | `generate.py`, `filters.py` — 7 scripts and code-switching detection. **Transliteration is not implemented.** The slice is small and hand-authored. |
| 1C data versioning, lineage, licensing | **partial** | `lineage.py`, `docs/lineage.md` — per-record provenance built and enforced. **No dataset-level versioning and no licence tracking**; see the "Where the policy stops" section. |

## Substitutions from the brief's stack

- **fastText → script detection.** `lid.176.bin` is a 126 MB download that makes
  a fresh clone fail offline and puts a third-party file server in CI. Script
  detection is the narrower capability, and every language claim in this repo
  is stated as a claim about script. It cannot separate Spanish from
  Portuguese.
- **`llm-client-kit` → unused.** The dependency exists to call a model, and no
  model is called here. Importing it to look consistent with the brief would
  add a dependency the code does not exercise.
- **`llm-eval-harness` → unused.** Its `stats.py` provides bootstrap confidence
  intervals for comparing two systems on shared samples. Every number in this
  repo is a **census** of a deterministic corpus rather than a sample estimate
  — recall is 63/108, not an average over draws — so there is no sampling
  distribution to bootstrap. A CI around an exact count would be decoration.
