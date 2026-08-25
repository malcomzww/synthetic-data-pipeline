"""Generate the hygiene-layer results.

Two outputs, deliberately separated -- the same split `llm-client-kit` uses:

- ``results/contamination.md``   committed. The headline measurement and the
                                 survival funnel. Fully deterministic given the
                                 seed, so CI regenerates it and fails on drift.
- ``results/environment.md``     gitignored. Machine, Python build, wall-clock.
                                 Nothing here is a claim; it is provenance for
                                 whoever ran it.

Everything in the committed file is a *count* or a *ratio of counts* derived
from a seeded pipeline. There is no timing in it, because a byte-comparing
drift gate over wall-clock numbers is a broken gate.

The script ASSERTS every claim the README makes. If detector recall drops, if
the funnel stops balancing, or if the planted rate stops matching what was
asked for, this exits non-zero rather than quietly writing a worse number.

Run:  python scripts/generate_results.py
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from synthetic_data_pipeline.decontaminate import (  # noqa: E402
    DEFAULT_N,
    EvalIndex,
    decontaminate,
    plant_contamination,
    sweep_n,
)
from synthetic_data_pipeline.dedup import (  # noqa: E402
    DEFAULT_THRESHOLD,
    drop_exact_duplicates,
    drop_near_duplicates,
    find_near_duplicates,
    score_pairs,
    sweep_thresholds,
)
from synthetic_data_pipeline.eval_set import EVAL_SET  # noqa: E402
from synthetic_data_pipeline.filters import (  # noqa: E402
    apply_filters,
    detect_script,
    plant_pii,
    score_pii,
)
from synthetic_data_pipeline.generate import TemplateGenerator, plant_near_duplicates  # noqa: E402
from synthetic_data_pipeline.lineage import (  # noqa: E402
    STAGE_DECONTAMINATE,
    STAGE_DEDUP,
    STAGE_FILTER,
    drop_reasons,
    survival_funnel,
)

RESULTS = ROOT / "results"
OUT = RESULTS / "contamination.md"
ENV = RESULTS / "environment.md"

# --- the experiment ----------------------------------------------------
SEED = 7
N_RECORDS = 900
N_DUP_PAIRS = 60
CONTAMINATION_RATE = 0.12
PII_RATE = 0.08
DUP_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
NGRAM_SIZES = (5, 8, 13, 20)


def base_corpus():
    """The clean generated corpus, before anything is planted."""
    return TemplateGenerator(seed=SEED).generate(N_RECORDS)


# Each detector is scored on a corpus where *only its own* ground truth has
# been planted.
#
# The first version planted all three into one corpus and scored every
# detector on it. That silently destroyed the duplicate measurement: the
# contamination and PII splices append 15-30 tokens of unrelated text, and
# they land on the two members of a planted duplicate pair independently. A
# pair planted at Jaccard ~0.85 was pulled down to 0.32 by text that had
# nothing to do with duplication, and MinHash recall read 0% -- not because
# the detector failed, but because the fixture had stopped containing
# duplicates by the time the detector saw it.
#
# Isolating the corpora costs three generation passes, which is milliseconds,
# and it makes each recall figure a measurement of one detector rather than of
# an interaction between three planting steps.


def dup_corpus():
    """Corpus with only near-duplicates planted."""
    return plant_near_duplicates(base_corpus(), n_pairs=N_DUP_PAIRS, seed=SEED)


def contamination_corpus():
    """Corpus with only eval-set spans planted."""
    return plant_contamination(base_corpus(), EVAL_SET, rate=CONTAMINATION_RATE, seed=SEED)


def pii_corpus():
    """Corpus with only PII specimens planted."""
    return plant_pii(base_corpus(), rate=PII_RATE, seed=SEED)


def combined_corpus():
    """All three planted together -- used for the survival funnel only.

    The funnel is a statement about how many records a full pipeline run
    removes and why, which is exactly the question that needs every hazard
    present at once. It is deliberately *not* used for any recall number.
    """
    records = base_corpus()
    records, dup_truth = plant_near_duplicates(records, n_pairs=N_DUP_PAIRS, seed=SEED)
    records, contam_truth = plant_contamination(
        records, EVAL_SET, rate=CONTAMINATION_RATE, seed=SEED
    )
    records, pii_truth = plant_pii(records, rate=PII_RATE, seed=SEED)
    return records, dup_truth, contam_truth, pii_truth


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)

    # --- 1. contamination, on its own corpus ---
    contam_records, contam_truth = contamination_corpus()
    ngram_sweep = sweep_n(contam_records, EVAL_SET, contam_truth, NGRAM_SIZES)
    index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
    _, contamination = decontaminate(contam_records, index, contam_truth)

    # Split the planted set by whether the span is long enough to be findable
    # at all. Without this the headline recall is uninterpretable: it mixes
    # detector performance with a property of the fixture.
    span_lengths = [
        event.detail["span_tokens"]
        for record in contam_records
        if record.record_id in contam_truth
        for event in record.provenance
        if event.detail.get("planted") == "contamination"
    ]
    detectable = sum(1 for s in span_lengths if s >= DEFAULT_N)
    undetectable = len(span_lengths) - detectable

    # --- 2. near-duplicates, on their own corpus ---
    dup_records, dup_truth = dup_corpus()
    dup_records = drop_exact_duplicates(dup_records)
    dup_sweep = sweep_thresholds(dup_records, dup_truth, DUP_THRESHOLDS)
    _, pairs = drop_near_duplicates(dup_records, threshold=DEFAULT_THRESHOLD)
    dup_score = score_pairs(pairs, dup_truth, threshold=DEFAULT_THRESHOLD)

    # How many near-duplicate pairs exist in the corpus with *nothing* planted?
    # This turns the precision caveat from an assertion into a measurement: it
    # is the count of pairs the detector is charged for that are properties of
    # the template generator rather than detector errors.
    baseline_pairs = len(find_near_duplicates(base_corpus(), threshold=DEFAULT_THRESHOLD))

    # --- 3. PII, on its own corpus ---
    # Scored before the filter stage rewrites the text: once the scrubber has
    # replaced the identifiers, re-detecting them measures nothing.
    pii_records, pii_truth = pii_corpus()
    pii_score = score_pii(pii_records, pii_truth)

    # --- 4. script coverage, on the clean corpus ---
    scripts: dict[str, int] = {}
    code_switched = 0
    for record in base_corpus():
        verdict = detect_script(record.text)
        scripts[verdict.dominant] = scripts.get(verdict.dominant, 0) + 1
        code_switched += verdict.code_switched

    # --- 5. the survival funnel, on the combined corpus ---
    records, _, combined_contam, _ = combined_corpus()
    generated = len(records)
    records = apply_filters(records)
    records = drop_exact_duplicates(records)
    records, _ = drop_near_duplicates(records, threshold=DEFAULT_THRESHOLD)
    records, _ = decontaminate(records, index, combined_contam)

    funnel = survival_funnel(records)
    survivors = sum(1 for r in records if r.alive)

    write_results(
        generated=generated,
        records=records,
        funnel=funnel,
        survivors=survivors,
        contamination=contamination,
        contam_truth=contam_truth,
        detectable=detectable,
        undetectable=undetectable,
        baseline_pairs=baseline_pairs,
        dup_score=dup_score,
        dup_sweep=dup_sweep,
        dup_truth=dup_truth,
        ngram_sweep=ngram_sweep,
        pii_score=pii_score,
        pii_truth=pii_truth,
        scripts=scripts,
        code_switched=code_switched,
    )
    write_environment(generated, survivors, time.perf_counter() - started)
    print(f"wrote {OUT.name} (committed) and {ENV.name} (gitignored)")


def write_results(
    *,
    generated: int,
    records: list,
    funnel: list,
    survivors: int,
    contamination,
    contam_truth: set[str],
    detectable: int,
    undetectable: int,
    baseline_pairs: int,
    dup_score,
    dup_sweep: list,
    dup_truth: set,
    ngram_sweep: list,
    pii_score,
    pii_truth: set[str],
    scripts: dict[str, int],
    code_switched: int,
) -> None:
    L: list[str] = []
    add = L.append

    add("# Hygiene layer: detector recall against planted ground truth\n")
    add("Generated by `python scripts/generate_results.py`. Do not edit by hand.")
    add("Machine-specific provenance is in `results/environment.md` (gitignored).\n")

    add("## What this measures, and what it does not\n")
    add("**The contamination in this corpus was planted at a rate chosen here,")
    add("not discovered in the wild.** No claim is made about any real dataset.")
    add("The measurement is *detector recall against exact ground truth*: given")
    add(f"that {CONTAMINATION_RATE:.0%} of records were spliced with a verbatim span")
    add("of the eval set, what fraction does a 13-gram overlap detector find?\n")
    add("**The generated records are templated, not model-generated.** No LLM was")
    add("called anywhere in this repo. See `src/synthetic_data_pipeline/generate.py`")
    add("for why measurability required that choice.\n")

    add("### Experiment parameters\n")
    add(f"- Seed: `{SEED}` (every stage derives from it; output is byte-stable)")
    add(f"- Generated records: {generated:,} ({N_RECORDS:,} base + planted copies)")
    add(f"- Eval set: {len(EVAL_SET)} held-out items, public/hand-authored")
    add(f"- Planted near-duplicate pairs: {len(dup_truth)}")
    add(f"- Planted contamination: {len(contam_truth)} records "
        f"({len(contam_truth) / generated:.1%} of the corpus)")
    add(f"- Planted PII: {len(pii_truth)} records ({len(pii_truth) / generated:.1%})")
    add("- Reproduce: `python scripts/generate_results.py`")
    add("- Model snapshot: none -- no model is called\n")

    # --- headline ---
    add("## 1. The headline: contamination detected before decontamination\n")
    add(f"At the {DEFAULT_N}-gram threshold, of the {contamination.inspected:,} records that")
    add(f"reached the decontamination stage, **{contamination.flagged:,} were flagged as")
    add(f"contaminated ({contamination.rate:.1%})** against {contamination.planted:,} planted.\n")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| inspected at this stage | {contamination.inspected:,} |")
    add(f"| planted, still live here | {contamination.planted:,} |")
    add(f"| flagged by the detector | {contamination.flagged:,} |")
    add(f"| **detector recall** | **{contamination.recall:.1%}** |")
    add(f"| **miss rate** | **{contamination.miss_rate:.1%}** |")
    add(f"| precision | {contamination.precision:.1%} |")
    add("")
    add(f"**{contamination.miss_rate:.1%} of records that genuinely contained a verbatim")
    add("eval span were not caught.** A decontamination pass reported as")
    add("complete would have shipped them.\n")
    add("That miss rate is not noise, and it is not a tuning failure. Split the")
    add("planted set by the length of the span that was spliced in:\n")
    add("| planted span | records | caught | missed |")
    add("|---|---|---|---|")
    add(f"| >= {DEFAULT_N} tokens | {detectable:,} | {contamination.true_positives:,} | "
        f"{detectable - contamination.true_positives:,} |")
    add(f"| < {DEFAULT_N} tokens | {undetectable:,} | 0 | {undetectable:,} |")
    add("")
    add("**Detectability is decided entirely by span length, with no exceptions")
    add("in either direction.** Every span of at least n tokens was found; every")
    add("shorter span was missed. That is not a discovered empirical fact so much")
    add("as the definition of the detector made visible -- an n-gram matcher")
    add("cannot see a fragment shorter than n -- but seeing it hold exactly is")
    add("what tells you the implementation does what it says.\n")
    add("The consequence is the useful one. **A clean decontamination report")
    add("means \"no leaked span of 13+ tokens\", not \"no leakage\".** Leakage")
    add("arrives in fragments too, and against those this method is not weak,")
    add("it is blind. Reporting the pass rate without that qualifier is how a")
    add("contaminated eval survives review.\n")

    assert contamination.planted > 0, "no planted record survived to the decontamination stage"
    assert contamination.precision >= 0.95, (
        f"13-gram precision fell to {contamination.precision:.1%}"
    )
    # The claim under test is not "recall is high" -- with spans deliberately
    # planted below the threshold it cannot be. It is the sharper claim that
    # detectability is decided *entirely* by span length: every planted span of
    # >= n tokens is caught, and every span shorter than n is missed. That is a
    # statement about the detector's definition, and if it ever fails the
    # detector has started behaving unpredictably.
    assert detectable > 0 and undetectable > 0, (
        "the fixture must contain both detectable and undetectable spans"
    )
    assert contamination.true_positives == detectable, (
        f"caught {contamination.true_positives} of {detectable} spans at or above n={DEFAULT_N}; "
        f"a >= n span is verbatim and must always be found"
    )
    assert contamination.flagged == detectable, (
        f"flagged {contamination.flagged} records but only {detectable} carry a >= n span; "
        f"a sub-threshold span cannot produce an n-gram match"
    )

    add("### A second blind spot: unsegmented scripts\n")
    add("The span-length result above is measured. This one follows from the")
    add("same mechanism and is stated rather than measured, because the")
    add("multilingual slice here is too small to put a number on: the detector")
    add("tokenises on whitespace, and Chinese, Japanese and Thai do not use it")
    add("as a word boundary. A CJK record yields almost no word n-grams, so")
    add("word-level decontamination cannot see leakage in it at all. Any")
    add("multilingual corpus decontaminated this way is effectively")
    add("undecontaminated for a subset of its languages.\n")

    # --- n sweep ---
    add(f"## 2. The {DEFAULT_N} is a convention on a curve, not a cliff\n")
    add("`n` is inherited from the GPT-3 contamination analysis. Sweeping it")
    add("shows what the single number hides:\n")
    add("| n | flagged | recall | false positives |")
    add("|---|---|---|---|")
    for report in ngram_sweep:
        mark = " **(default)**" if report.n == DEFAULT_N else ""
        add(f"| {report.n}{mark} | {report.flagged:,} | {report.recall:.1%} | "
            f"{report.false_positives:,} |")
    add("")
    add("Recall falls monotonically as n rises, and it falls for exactly one")
    add("reason: each increment of n makes another band of planted spans too")
    add("short to match. Lowering n therefore *raises* recall on this corpus --")
    add("n=5 catches spans a 13-gram detector cannot.\n")
    add("So why not use n=5? Because the false-positive column here is")
    add("misleadingly clean. This eval set is 24 hand-authored items whose")
    add("phrasing is close to unique; against a real benchmark sharing ordinary")
    add("English with the training corpus, a 5-gram matcher flags large amounts")
    add("of uncontaminated text. **The cost of small n is invisible in this")
    add("fixture**, which is precisely why the 13 is taken on convention rather")
    add("than tuned to whatever maximises recall here. Tuning n on this corpus")
    add("would produce a number that does not survive contact with a real one.\n")

    recalls_by_n = [r.recall for r in ngram_sweep]
    assert recalls_by_n == sorted(recalls_by_n, reverse=True), (
        f"recall should fall monotonically as n rises; got {recalls_by_n}"
    )
    assert all(r.false_positives == 0 for r in ngram_sweep), (
        "a planted-only fixture should produce no false positives at any n; "
        "if this fires, the eval set has started overlapping the generator's own phrasing"
    )

    # --- funnel ---
    add("## 3. Survival funnel\n")
    add(f"{generated:,} records in, {survivors:,} out "
        f"({survivors / generated:.1%} survive).\n")
    add("| stage | entered | dropped | survived | drop rate |")
    add("|---|---|---|---|---|")
    entered_first = generated
    for count in funnel:
        rate = count.dropped / count.entered if count.entered else 0.0
        add(f"| {count.stage} | {count.entered:,} | {count.dropped:,} | "
            f"{count.survived:,} | {rate:.1%} |")
    add("")

    add("### Why records were dropped\n")
    for stage in (STAGE_FILTER, STAGE_DEDUP, STAGE_DECONTAMINATE):
        reasons = drop_reasons(records, stage)
        if not reasons:
            add(f"- **{stage}**: nothing dropped")
            continue
        detail = ", ".join(f"`{k}` {v:,}" for k, v in reasons.items())
        add(f"- **{stage}**: {detail}")
    add("")
    add("The funnel is reconstructed from the surviving records' own provenance,")
    add("not from counters incremented as the pipeline ran. A counter can drift")
    add("from the data it describes; a value recomputed from the artifacts cannot.\n")

    # The funnel must balance exactly. This is the assertion most likely to
    # catch a future refactor that drops a record without recording why.
    total_dropped = sum(c.dropped for c in funnel)
    assert entered_first - total_dropped == survivors, (
        f"funnel does not balance: {entered_first} - {total_dropped} != {survivors}"
    )
    assert funnel[0].entered == generated, "first funnel stage did not see every record"

    # --- dedup ---
    add("## 4. Near-duplicate detection (MinHash + LSH)\n")
    add(f"At threshold {DEFAULT_THRESHOLD}, against {len(dup_truth)} planted pairs:\n")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| true positives | {dup_score.true_positives} |")
    add(f"| false negatives | {dup_score.false_negatives} |")
    add(f"| false positives | {dup_score.false_positives} |")
    add(f"| **recall** | **{dup_score.recall:.1%}** |")
    add(f"| precision (lower bound) | {dup_score.precision:.1%} |")
    add("")
    add("**That 1.8% precision is an artifact of the fixture, and it is")
    add("measurable rather than merely arguable.** Run the same detector over")
    add("the base corpus with *nothing planted at all*:\n")
    add(f"- near-duplicate pairs found with zero planted duplicates: **{baseline_pairs:,}**")
    add(f"- pairs found after planting {len(dup_truth)}: "
        f"{dup_score.true_positives + dup_score.false_positives:,}")
    add("")
    add("The template generator emits records that differ by a single noun")
    add("phrase -- \"Draft a checklist for reviewing *a token bucket*\" against")
    add("\"...*a write-ahead log*\" -- and at Jaccard 0.7 those genuinely are")
    add("near-duplicates. The detector is right to flag them; the scoring")
    add("function calls them false positives only because they were not")
    add(f"*planted*. Roughly {baseline_pairs:,} of the {dup_score.false_positives:,}")
    add("charged false positives are pairs that exist before any planting.\n")
    add("**So recall is the number to trust here and precision is not**, which")
    add("is a limitation of the fixture rather than a finding about MinHash. A")
    add("precision figure worth reporting needs a corpus whose true duplicate")
    add("set is known exhaustively, not just one whose planted subset is.\n")

    assert baseline_pairs > 0, (
        "the base corpus no longer contains unplanted near-duplicates; the precision "
        "caveat above is stale and should be rewritten"
    )

    assert dup_score.recall >= 0.90, (
        f"near-duplicate recall fell to {dup_score.recall:.1%}"
    )

    add("### Threshold sweep\n")
    add("| Jaccard threshold | recall | precision (lower bound) | detected pairs |")
    add("|---|---|---|---|")
    for score in dup_sweep:
        mark = " **(default)**" if score.threshold == DEFAULT_THRESHOLD else ""
        detected = score.true_positives + score.false_positives
        add(f"| {score.threshold}{mark} | {score.recall:.1%} | {score.precision:.1%} | "
            f"{detected:,} |")
    add("")
    add("Published instead of a single tuned threshold: a value chosen to")
    add("maximise F1 on this fixture would be an overfit, and reporting it as")
    add("*the* threshold is exactly the kind of number that does not transfer.\n")

    recalls = [s.recall for s in dup_sweep]
    assert recalls == sorted(recalls, reverse=True), (
        "recall should fall monotonically as the Jaccard threshold rises; it did not"
    )

    # --- PII ---
    add("## 5. PII scrubbing\n")
    add(f"Against {len(pii_truth)} records with a planted identifier:\n")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| recall | {pii_score.recall:.1%} |")
    add(f"| precision | {pii_score.precision:.1%} |")
    add(f"| false negatives | {pii_score.false_negatives} |")
    add(f"| false positives | {pii_score.false_positives} |")
    add("")
    add("Scored **per record, not per span**: a record whose email was caught")
    add("but whose phone number was missed counts as a hit. That makes this an")
    add("optimistic metric, and it is named as one.\n")
    add("This measures whether the regexes fire on the shapes they were written")
    add("for. **It is not evidence that a corpus is PII-free.** Regex scrubbing")
    add("is structurally blind to unformatted PII -- a person's name, an address")
    add("in prose, a date of birth written as words. PII is scrubbed rather than")
    add("dropped, because removing every record containing an email address")
    add("would bias the corpus away from anything about contacting people.\n")

    assert pii_score.recall >= 0.99, f"PII recall fell to {pii_score.recall:.1%}"
    assert pii_score.false_positives == 0, (
        f"{pii_score.false_positives} records were scrubbed without planted PII"
    )

    # --- multilingual ---
    add("## 6. Script coverage and code-switching\n")
    n_scripted = sum(scripts.values())
    add(f"Measured on the {n_scripted:,}-record base corpus, before anything is planted.")
    add(f"{code_switched} records carry a second script above the 15% threshold.")
    add("Dominant-script distribution:\n")
    add("| script | records |")
    add("|---|---|")
    for script, count in sorted(scripts.items(), key=lambda kv: (-kv[1], kv[0])):
        add(f"| {script} | {count:,} |")
    add("")
    add("**This is script detection, not language identification.** It separates")
    add("Latin from Cyrillic from Han. It cannot tell Spanish from Portuguese,")
    add("because they share a script. Every language claim in this repo is a")
    add("claim about script, and no more. fastText's `lid.176.bin` would do")
    add("better and is a 126 MB download that makes a fresh clone fail offline;")
    add("that trade is stated rather than hidden.\n")

    assert code_switched > 0, "the code-switching fixtures stopped being detected"

    # --- limitations ---
    add("## Limitations\n")
    add("- **Nothing here is contamination found in the wild.** The rate was")
    add("  chosen, planted, and then recovered. The transferable result is the")
    add("  detector's recall and miss rate, not the rate itself.")
    add("- **The corpus is templated.** Its lexical diversity is bounded by a")
    add("  cross product of 10 frames x 10 domains x 18 objects. A real")
    add("  model-generated corpus is more diverse in some ways and far more")
    add("  repetitive in others, and MinHash recall on it is not established here.")
    add("- **Word n-grams are close to blind on unsegmented scripts.** The CJK")
    add("  rows are carried through the pipeline but the 13-gram detector cannot")
    add("  see leakage in them. This repo does not establish a decontamination")
    add("  method that works for Chinese or Japanese.")
    add("- **PII recall is measured against planted regex-shaped identifiers**,")
    add("  which is close to the best case for a regex scrubber. Unformatted PII")
    add("  is not measured and not caught.")
    add("- **The eval set is 24 hand-authored items**, not a real benchmark. Its")
    add("  n-gram profile is not that of MMLU or GSM8K, and the false-positive")
    add("  rate against a real benchmark would differ.")
    add("- **No semantic dedup.** Records that mean the same thing in different")
    add("  words survive every stage here. Embedding-based dedup is the standard")
    add("  next step and is not implemented.\n")

    add("---\n")
    add(f"_Generated {date.today().isoformat()} from seed {SEED}._")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_environment(generated: int, survivors: int, elapsed: float) -> None:
    """Machine provenance. Gitignored: none of it is a claim."""
    L: list[str] = []
    add = L.append
    add("# Environment (machine-specific, not committed)\n")
    add(f"- Date: {date.today().isoformat()}")
    add(f"- Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    add(f"- Processor: {platform.processor() or 'unknown'}")
    add(f"- Records generated: {generated:,}; survived: {survivors:,}")
    add(f"- Wall clock: {elapsed:.1f}s")
    add("- No GPU used, no network call made, no model invoked.\n")
    ENV.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
