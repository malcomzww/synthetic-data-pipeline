# Provenance policy

The rule this repository enforces: **a record carries its own history, and a
record is never deleted.**

Both halves are load-bearing, and both cost something. This document says what
the policy is, what it buys, what it costs, and where it stops.

## The policy

### 1. Provenance is a field, not a side table

Every `Record` holds a `provenance` tuple. Each stage appends an event to it
and returns a *new* record; nothing is mutated in place.

The alternative — a side table keyed by record id, written by each stage — is
cheaper and is what most pipelines start with. It decays the first time
somebody filters a shard with a one-off script and does not update the table,
and that always happens eventually. When provenance is a field, a record
cannot be moved, copied, subsetted, or shuffled into a training mix without
its history coming along. The invariant survives handling by people who have
never read this document.

**Cost:** every record is larger, and the history only grows. Storage is the
cheap part of a training pipeline; an unexplainable corpus is not.

### 2. Dropped records are kept, not deleted

A record removed by any stage gets `dropped_by` set and stays in the list. The
survival funnel is then recomputed from the records themselves.

The reason is that the interesting question is almost never "how many records
survived?" It is "what did we throw away, and was that right?" A pipeline that
deletes its rejects can answer the first and not the second. It also means the
funnel cannot silently disagree with the data: counters incremented as a
pipeline runs can drift away from what actually happened, while a value
derived from the artifacts cannot.

**Cost:** peak memory holds the whole corpus including rejects. For corpora
larger than memory this policy needs a different implementation — dropped
records streamed to a side file — but the same invariant.

### 3. Every drop states a reason

Dropping requires a reason string, which becomes `drop:<reason>` in the
provenance and is what `drop_reasons()` aggregates. A funnel row saying "565
records dropped" is a number; "565 dropped, `near_duplicate` 565" is a finding.

### 4. Re-dropping is an error, not an overwrite

`Record.dropped()` raises if the record was already dropped. If a second stage
sees an already-dropped record then the stage ordering is wrong, and the
*first* cause is the true one. Overwriting it would make the funnel attribute
the loss to the wrong stage — a bug that produces a plausible-looking report,
which is the worst kind.

### 5. Stage names are a closed set

`ProvenanceEvent` rejects any stage outside `STAGES`. A typo like `"dedupe"`
becomes an exception rather than a silent new category that splits the funnel
into two rows nobody notices.

## What a record's history looks like

A record that survived the whole pipeline:

| stage | action | detail |
|---|---|---|
| `generate` | `template` | `source=templated`, `generator=TemplateGenerator`, `seed=7`, `model=None` |
| `filter` | `keep` | `tokens=27`, `script=latin`, `code_switched=False` |
| `dedup` | `keep` | `check=exact` |
| `dedup` | `keep` | `check=near`, `threshold=0.7` |
| `decontaminate` | `keep` | `n=13` |

One that did not:

| stage | action | detail |
|---|---|---|
| `generate` | `template` | `source=templated`, `seed=7`, `model=None` |
| `filter` | `keep` | `tokens=31`, `script=latin` |
| `dedup` | `keep` | `check=exact` |
| `dedup` | `drop:near_duplicate` | `duplicate_of=3f2a...`, `threshold=0.7` |

The second record names the record that displaced it and the threshold that
decided it. Both are needed to re-litigate the decision later: a different
threshold would have kept it, and the survivor is where its content went.

## The generation label

Every record generated here carries `source="templated"` and `model=None`.

**No language model was called anywhere in this repository.** The generation
step is templated recombination standing in for a self-instruct style LLM
loop. This is asserted by a test, not just documented, because it is the claim
most likely to be misread by someone skimming: a "synthetic data pipeline"
that reports a contamination rate invites the assumption that a model produced
the data and the rate was discovered in the wild. Neither is true here.

If an LLM-backed generator is added later it implements the same `Generator`
protocol and writes its own model identifier into that same field. Nothing
downstream changes, and the label distinguishes the two corpora forever.

## Where the policy stops

Things this provenance model does **not** give you, listed because a lineage
document that only lists strengths is marketing:

- **No content hash of the record at each stage.** You can see that the filter
  stage scrubbed PII, but not what the text was before. Reconstructing the
  pre-scrub text is impossible by design — storing it would mean keeping the
  identifiers the stage exists to remove.
- **No cross-record lineage for merges.** Records are only ever dropped or
  rewritten in place, never combined. A pipeline that fused two records into
  one would need a genuine DAG here, not a list.
- **No dataset-level versioning.** Each record knows its own history; nothing
  records "this corpus is v3, built from config X". That belongs in a manifest
  alongside the JSONL and is not implemented.
- **No licence tracking.** The brief lists licensing under data versioning, and
  it is not modelled. Every input here is either hand-authored in this repo or
  generated by it, so there is nothing to track — which means the design was
  never tested against the case that matters, a corpus mixing sources under
  different terms.
- **`plant_contamination` and `plant_pii` deliberately break the content-id
  invariant.** They mutate text while keeping the record id, because ground
  truth is keyed on that id. They are test fixtures, not pipeline stages, and
  the provenance event records the splice. This is the one place in the
  codebase where id and content diverge.
