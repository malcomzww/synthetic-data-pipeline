"""N-gram decontamination against a held-out eval set.

This module answers the repo's one question:

    What fraction of a generated set is contaminated against the eval set
    before decontamination?

**Read this before reading any number produced here.** The contamination is
*planted* at a rate chosen by the experiment. Nothing in this repo discovers
contamination in the wild, and no claim is made about any real corpus. What is
measured is **detector recall against exact ground truth**: given that N% was
planted, what fraction does a 13-gram overlap detector actually find? The miss
rate is the more useful half of that answer, and it is reported.

**Why 13-gram.** The threshold comes from the GPT-3 paper's contamination
analysis and has been carried forward by most large-corpus work since. It is a
convention rather than a derived optimum, and this module treats it that way:
`n` is a parameter, the sweep across `n` is reported, and the 13 is labelled as
inherited rather than discovered. The intuition behind it is that 13
consecutive words co-occurring by chance in unrelated text is vanishingly
unlikely, while 5 or 6 is routine -- so short n over-flags and long n misses
paraphrased leakage.

**Word n-grams here, character shingles in dedup.** The two stages tokenise
differently on purpose. Dedup asks "is this the same record?", which is a
fuzzy, script-agnostic question that character shingles answer well.
Decontamination asks "did a verbatim span of the eval set leak in?", which is a
question about words, and where the 13-gram convention is defined. Using
character n-grams here would make the threshold meaningless.

The consequence is honest and worth stating: for the CJK rows in the
multilingual slice, whitespace tokenisation produces almost no word n-grams,
so this detector is close to blind on them. See `docs/lineage.md`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .lineage import STAGE_DECONTAMINATE, Record

DEFAULT_N = 13

# Words as runs of letters/digits/underscore. Punctuation is a separator, not
# a token, so "cache," and "cache" produce the same n-gram -- a formatting
# difference must not let contamination through.
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenise(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def ngrams(text: str, n: int = DEFAULT_N) -> set[str]:
    """Word n-grams of ``text``, joined by single spaces.

    A text with fewer than ``n`` tokens yields the empty set. That is the
    correct behaviour, not a gap: a 6-word record cannot contain a 13-gram, so
    it is not detectably contaminated at this threshold, and pretending
    otherwise by padding or backing off to a shorter n would silently change
    the threshold the number is reported under.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tokens = tokenise(text)
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass(frozen=True, slots=True)
class EvalIndex:
    """The eval set, indexed as an n-gram set for O(1) membership.

    Built once and reused across every record. The naive alternative --
    comparing each generated record against each eval item -- is O(G x E) and
    turns a minute into an hour on a corpus of any size.
    """

    n: int
    grams: frozenset[str]
    n_items: int

    @classmethod
    def build(cls, eval_texts: Iterable[str], *, n: int = DEFAULT_N) -> EvalIndex:
        texts = list(eval_texts)
        grams: set[str] = set()
        for text in texts:
            grams |= ngrams(text, n)
        return cls(n=n, grams=frozenset(grams), n_items=len(texts))

    def overlap(self, text: str) -> set[str]:
        """The n-grams ``text`` shares with the eval set."""
        return ngrams(text, self.n) & self.grams

    def is_contaminated(self, text: str) -> bool:
        """One shared n-gram is enough.

        Requiring several would raise precision but is not what the convention
        says, and the whole point of choosing n large is that a single
        coincidental match is already implausible. Moving the bar to 2+ would
        be a second undeclared threshold hiding inside the first.
        """
        return bool(self.overlap(text))


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """The repo's headline measurement.

    ``rate`` is the fraction of the *inspected* corpus flagged as contaminated.
    ``recall`` and ``miss_rate`` compare that against the planted ground truth
    and are the numbers that mean something, since the planted rate was chosen
    rather than discovered.
    """

    n: int
    inspected: int
    flagged: int
    planted: int
    true_positives: int
    false_positives: int

    @property
    def rate(self) -> float:
        return self.flagged / self.inspected if self.inspected else 0.0

    @property
    def planted_rate(self) -> float:
        return self.planted / self.inspected if self.inspected else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.planted if self.planted else 1.0

    @property
    def miss_rate(self) -> float:
        """The number that is easy to omit and most worth reporting."""
        return 1.0 - self.recall

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0


def plant_contamination(
    records: Sequence[Record],
    eval_texts: Sequence[str],
    *,
    rate: float,
    seed: int = 0,
    n: int = DEFAULT_N,
    min_span: int = 6,
    max_span: int = 24,
) -> tuple[list[Record], set[str]]:
    """Splice eval-set spans into a chosen fraction of the corpus.

    Returns the modified corpus and the ground-truth set of contaminated
    record ids.

    The splice takes a contiguous span from an eval item and appends it to the
    record's response, which is what real leakage looks like: a model that
    memorised a benchmark item reproduces a run of it verbatim inside
    otherwise novel text.

    **``min_span`` defaults below ``n``, and that is the point.** The first
    version of this fixture drew every span at ``n`` tokens or longer, which
    made the measurement very nearly tautological: a 13-gram detector cannot
    miss a verbatim 13-token span, so recall came out at exactly 100% and the
    reported "miss rate" of 0% measured the fixture rather than the detector.

    Real leakage is not so obliging. A model reproduces a remembered *phrase*
    as often as a remembered sentence, and a span shorter than ``n`` is
    invisible to an n-gram detector by construction. Spanning 6 to 24 tokens
    puts genuinely undetectable cases in the corpus, so the resulting recall
    is a real number with a real miss rate attached -- and the misses have a
    stateable cause rather than being an artifact.

    Record ids are **not** recomputed after splicing. The id must stay stable
    so ground truth keyed on it survives the mutation; the provenance event
    records that the text changed. This is the one place in the pipeline where
    id and content deliberately diverge, and it is a test fixture, not a
    pipeline stage.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0,1], got {rate}")
    if not eval_texts:
        raise ValueError("cannot plant contamination from an empty eval set")
    if min_span < 1 or max_span < min_span:
        raise ValueError(f"need 1 <= min_span <= max_span, got {min_span}..{max_span}")

    import random

    rng = random.Random(seed)
    usable = [t for t in eval_texts if len(tokenise(t)) >= min_span]
    if not usable:
        raise ValueError(
            f"no eval text has {min_span} tokens; cannot plant a span of that length"
        )

    target = int(round(len(records) * rate))
    chosen = set(rng.sample(range(len(records)), min(target, len(records))))

    out: list[Record] = []
    truth: set[str] = set()
    for i, record in enumerate(records):
        if i not in chosen:
            out.append(record)
            continue
        source = rng.choice(usable)
        tokens = tokenise(source)
        # Span length is drawn across a range that straddles ``n``, so the
        # corpus contains both detectable and undetectable leakage.
        span_len = rng.randint(min_span, min(len(tokens), max_span))
        start = rng.randrange(0, len(tokens) - span_len + 1)
        span = " ".join(tokens[start : start + span_len])
        spliced = record.response + " " + span
        out.append(
            record.__class__(
                record_id=record.record_id,
                instruction=record.instruction,
                response=spliced,
                language=record.language,
                provenance=record.provenance,
                dropped_by=record.dropped_by,
            ).with_event(
                "generate",
                "splice",
                planted="contamination",
                span_tokens=span_len,
                eval_source_tokens=len(tokens),
            )
        )
        truth.add(record.record_id)
    return out, truth


def decontaminate(
    records: Sequence[Record],
    index: EvalIndex,
    planted: set[str] | None = None,
) -> tuple[list[Record], ContaminationReport]:
    """Drop every live record sharing an n-gram with the eval set.

    Only live records are inspected, so the reported contamination rate is a
    rate over what *reached* this stage, not over the original generation. The
    two differ once dedup has run, and conflating them would let a change in
    the dedup threshold silently move the headline number.
    """
    planted = planted or set()
    out: list[Record] = []
    inspected = flagged = tp = fp = 0

    for record in records:
        if not record.alive:
            out.append(record)
            continue
        inspected += 1
        hits = index.overlap(record.text)
        if hits:
            flagged += 1
            if record.record_id in planted:
                tp += 1
            else:
                fp += 1
            # Store one exemplar n-gram, not all of them: the provenance is an
            # audit trail, and a record that leaked 400 grams would otherwise
            # bloat the JSONL by more than the record itself.
            out.append(
                record.dropped(
                    STAGE_DECONTAMINATE,
                    "ngram_overlap",
                    n=index.n,
                    overlap_count=len(hits),
                    example=sorted(hits)[0],
                )
            )
        else:
            out.append(record.kept(STAGE_DECONTAMINATE, n=index.n))

    planted_inspected = sum(
        1 for r in records if r.alive and r.record_id in planted
    )
    return out, ContaminationReport(
        n=index.n,
        inspected=inspected,
        flagged=flagged,
        planted=planted_inspected,
        true_positives=tp,
        false_positives=fp,
    )


def sweep_n(
    records: Sequence[Record],
    eval_texts: Sequence[str],
    planted: set[str],
    ns: Sequence[int],
) -> list[ContaminationReport]:
    """Detector behaviour across n-gram sizes.

    The point of the sweep is to show the 13 is a convention on a curve, not a
    cliff -- and to make visible that short n flags records that were never
    planted, which is the failure mode a single-threshold report hides.

    **Must be given the corpus as it stood *before* decontamination.** Each
    call re-inspects only live records, so passing an already-decontaminated
    corpus means the contaminated records have all been dropped and every row
    of the sweep reports zero -- which is what the first run of this function
    did. Guarding rather than documenting it: a sweep that silently reports
    all-zero looks like a finding ("no contamination at any n") when it is
    actually a caller error.
    """
    if planted and not any(r.alive and r.record_id in planted for r in records):
        raise ValueError(
            "no planted record is still live; sweep_n needs the corpus from before "
            "decontaminate() dropped them, otherwise every row reports zero"
        )
    return [
        decontaminate(records, EvalIndex.build(eval_texts, n=n), planted)[1] for n in ns
    ]
