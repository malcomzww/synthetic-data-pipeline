"""Exact and near-duplicate detection.

Two stages, in this order, because they answer different questions and the
cheap one shrinks the input to the expensive one:

- **Exact.** SHA-256 over the canonical text. Catches byte-identical repeats,
  which in a real generation run come from a model resampling the same
  completion. Free, and exactly correct.
- **Near.** MinHash + LSH over character shingles. Catches the paraphrases
  exact hashing cannot, at the cost of two tunable knobs and a recall that is
  no longer 100%.

The order matters for the funnel: reporting near-duplicate recall on a corpus
that still contains exact repeats would inflate it, since exact repeats are
the easy half of the problem.

**Why character shingles rather than word shingles.** Word shingles are the
common choice and are cheaper, but they are brittle across the multilingual
slice: Chinese and Japanese have no whitespace word boundaries, so a
whitespace tokeniser sees one enormous token and MinHash degenerates. Character
shingles are script-agnostic. The price is a larger shingle set per record and
a slower hash.

**Why the threshold is a parameter with a stated default rather than a tuned
constant.** 0.7 Jaccard on 5-character shingles is a starting point, not a
discovered optimum. The honest presentation is the precision/recall curve
across thresholds -- which `sweep_thresholds` produces -- because the right
operating point depends on whether the caller would rather lose good data or
keep duplicates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from .lineage import STAGE_DEDUP, Record

SHINGLE_SIZE = 5
NUM_PERM = 128
DEFAULT_THRESHOLD = 0.7

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Case-fold and collapse whitespace before shingling.

    Deliberately conservative: no punctuation stripping, no stemming, no
    Unicode NFKC. Every additional normalisation raises recall and lowers
    precision, and aggressive normalisation is how a dedup pass starts
    collapsing records that a human would call distinct. Case and whitespace
    are the two edits that are almost never semantic.
    """
    return _WHITESPACE.sub(" ", text.strip()).casefold()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams of the normalised text.

    A text shorter than ``size`` yields itself as a single shingle rather than
    an empty set, so that short records still hash to something and do not all
    collide into one empty-set group.
    """
    if size < 1:
        raise ValueError(f"shingle size must be >= 1, got {size}")
    norm = normalise(text)
    if len(norm) < size:
        return {norm} if norm else set()
    return {norm[i : i + size] for i in range(len(norm) - size + 1)}


def minhash(text: str, *, num_perm: int = NUM_PERM, size: int = SHINGLE_SIZE) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, size):
        m.update(sh.encode("utf-8"))
    return m


def jaccard(a: str, b: str, *, size: int = SHINGLE_SIZE) -> float:
    """True Jaccard on shingle sets. Used to verify LSH candidates.

    LSH returns *candidates*, not matches. Skipping this verification is the
    standard way a MinHash dedup pass ends up with precision well below what
    the threshold implies, because banding admits pairs below the threshold by
    design.
    """
    sa, sb = shingles(a, size), shingles(b, size)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


@dataclass(frozen=True, slots=True)
class DedupReport:
    """What the dedup stage did, in numbers the funnel can consume."""

    exact_dropped: int
    near_dropped: int
    near_pairs: frozenset[tuple[str, str]]

    @property
    def total_dropped(self) -> int:
        return self.exact_dropped + self.near_dropped


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical pair ordering so (x,y) and (y,x) compare equal."""
    return (a, b) if a <= b else (b, a)


def drop_exact_duplicates(records: Sequence[Record]) -> list[Record]:
    """Keep the first occurrence of each canonical text; drop the rest.

    "First" is by input order, which the generator makes deterministic. An
    arbitrary survivor would be equally valid statistically but would break the
    drift gate, so the deterministic choice is the one that is made.
    """
    seen: dict[str, str] = {}
    out: list[Record] = []
    for record in records:
        if not record.alive:
            out.append(record)
            continue
        key = normalise(record.text)
        if key in seen:
            out.append(
                record.dropped(STAGE_DEDUP, "exact_duplicate", duplicate_of=seen[key])
            )
        else:
            seen[key] = record.record_id
            out.append(record.kept(STAGE_DEDUP, check="exact"))
    return out


def find_near_duplicates(
    records: Sequence[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = NUM_PERM,
    size: int = SHINGLE_SIZE,
) -> set[tuple[str, str]]:
    """Candidate pairs from LSH, verified against true Jaccard.

    Returns canonically-ordered id pairs. Only live records participate: a
    record already dropped as an exact duplicate must not also be counted as a
    near duplicate, or the two stages double-count in the funnel.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0,1], got {threshold}")

    live = [r for r in records if r.alive]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: dict[str, MinHash] = {}
    texts: dict[str, str] = {}

    for record in live:
        # Content-addressed ids mean an id can legitimately repeat only if the
        # text is identical, which drop_exact_duplicates has already removed.
        # Guard anyway: LSH raises on a repeated key and a crash here would be
        # an obscure way to learn the stages ran out of order.
        if record.record_id in signatures:
            continue
        m = minhash(record.text, num_perm=num_perm, size=size)
        signatures[record.record_id] = m
        texts[record.record_id] = record.text
        lsh.insert(record.record_id, m)

    pairs: set[tuple[str, str]] = set()
    for rid, m in signatures.items():
        for other in lsh.query(m):
            if other == rid:
                continue
            pair = _ordered_pair(rid, other)
            if pair in pairs:
                continue
            if jaccard(texts[pair[0]], texts[pair[1]], size=size) >= threshold:
                pairs.add(pair)
    return pairs


def drop_near_duplicates(
    records: Sequence[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = NUM_PERM,
    size: int = SHINGLE_SIZE,
) -> tuple[list[Record], set[tuple[str, str]]]:
    """Drop the later member of each near-duplicate pair.

    Duplicate *groups* are resolved by keeping whichever member appears first
    in input order and dropping everything linked to it. Transitive chains
    (a~b, b~c, a!~c) therefore collapse to one survivor. That is the
    conservative choice -- it can drop a record that is not a duplicate of the
    survivor -- and it is the standard one, because the alternative leaves
    near-duplicate pairs in the output that the stage claims to have removed.
    """
    pairs = find_near_duplicates(records, threshold=threshold, num_perm=num_perm, size=size)

    order = {r.record_id: i for i, r in enumerate(records)}
    neighbours: dict[str, set[str]] = {}
    for a, b in pairs:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    # Greedy pass in input order: the first record of any group survives and
    # claims its neighbours. Deterministic given deterministic input order.
    dropped: dict[str, str] = {}
    for record in sorted(neighbours, key=lambda rid: order[rid]):
        if record in dropped:
            continue
        for neighbour in neighbours[record]:
            if neighbour not in dropped and order[neighbour] > order[record]:
                dropped[neighbour] = record

    out: list[Record] = []
    for record in records:
        if not record.alive:
            out.append(record)
        elif record.record_id in dropped:
            out.append(
                record.dropped(
                    STAGE_DEDUP,
                    "near_duplicate",
                    duplicate_of=dropped[record.record_id],
                    threshold=threshold,
                )
            )
        else:
            out.append(record.kept(STAGE_DEDUP, check="near", threshold=threshold))
    return out, pairs


@dataclass(frozen=True, slots=True)
class PairScore:
    """Precision/recall of detected pairs against a planted answer key."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_pairs(
    detected: set[tuple[str, str]],
    planted: set[tuple[str, str]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> PairScore:
    """Score detected pairs against ground truth.

    A caveat that belongs with the number rather than in a footnote: a "false
    positive" here is a pair the detector found that was not *planted*. The
    templated generator can produce genuinely similar records by chance -- two
    draws sharing a task frame and a domain differ only in the object -- and
    those are real near-duplicates that this metric charges against precision.
    So the precision figure is a **lower bound**, and recall is the number to
    trust. `score_pairs` cannot tell the difference; only inspection can.
    """
    detected_c = {_ordered_pair(*p) for p in detected}
    planted_c = {_ordered_pair(*p) for p in planted}
    tp = len(detected_c & planted_c)
    return PairScore(
        threshold=threshold,
        true_positives=tp,
        false_positives=len(detected_c - planted_c),
        false_negatives=len(planted_c - detected_c),
    )


def sweep_thresholds(
    records: Sequence[Record],
    planted: set[tuple[str, str]],
    thresholds: Sequence[float],
    *,
    num_perm: int = NUM_PERM,
    size: int = SHINGLE_SIZE,
) -> list[PairScore]:
    """Precision/recall across an operating range.

    Published instead of a single tuned threshold because the trade is the
    finding: a threshold chosen to maximise F1 on this corpus would be an
    overfit to a fixture, and reporting it as "the" threshold would be exactly
    the kind of number that does not transfer.
    """
    return [
        score_pairs(
            find_near_duplicates(records, threshold=t, num_perm=num_perm, size=size),
            planted,
            threshold=t,
        )
        for t in thresholds
    ]
