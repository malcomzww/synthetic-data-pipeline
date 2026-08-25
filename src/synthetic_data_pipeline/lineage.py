"""Provenance that travels with the record, not beside it.

The pipeline's whole value is being able to answer "why is this row in my
training set, and where did it come from?" months later. Two ways to build
that, and only one survives contact with reality:

1. A side table keyed by record id, written by each stage.
2. The provenance lives *inside* the record and every stage appends to it.

This module implements (2). The side table decays the moment someone filters a
shard with a one-off script and forgets to update it -- and that always
happens. When provenance is a field, a record cannot be moved, copied, or
shuffled into a training mix without carrying its own history along.

The cost is real: every record is bigger, and the history only grows. That is
the trade being made deliberately. Storage is cheap; an unexplainable training
corpus is not.

Nothing here mutates. Each stage returns a *new* record with one more event
appended, so a record that has been through the pipeline still contains its
own earlier states. That makes the survival funnel reconstructable from the
surviving records alone, and makes the dropped ones self-explaining.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# Stage names, fixed so a typo becomes an error rather than a silent new
# category in the funnel report.
STAGE_GENERATE = "generate"
STAGE_FILTER = "filter"
STAGE_DEDUP = "dedup"
STAGE_DECONTAMINATE = "decontaminate"

STAGES: tuple[str, ...] = (
    STAGE_GENERATE,
    STAGE_FILTER,
    STAGE_DEDUP,
    STAGE_DECONTAMINATE,
)


@dataclass(frozen=True, slots=True)
class ProvenanceEvent:
    """One thing that happened to a record.

    ``detail`` is free-form per stage but must be JSON-serialisable: the whole
    point is that this survives a round trip through JSONL on disk.
    """

    stage: str
    action: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {STAGES}")


@dataclass(frozen=True, slots=True)
class Record:
    """A single instruction-tuning example plus its history.

    ``instruction``/``response`` are the payload. Everything else exists to
    make the payload accountable.

    ``dropped_by`` is the stage that removed the record, or None if it is still
    alive. Dropped records are *kept*, not deleted -- a funnel you cannot audit
    is a funnel you cannot trust, and the interesting question is always "what
    did we throw away and was that right?"
    """

    record_id: str
    instruction: str
    response: str
    language: str = "en"
    provenance: tuple[ProvenanceEvent, ...] = ()
    dropped_by: str | None = None

    @property
    def alive(self) -> bool:
        return self.dropped_by is None

    @property
    def text(self) -> str:
        """The canonical text used by every content-addressed stage.

        Defined once, here, because dedup and decontamination must agree on
        what "the content" is. If one hashed only the instruction and the
        other hashed both fields, their numbers would not compose and the
        funnel would quietly lie.
        """
        return f"{self.instruction}\n{self.response}"

    def with_event(
        self,
        stage: str,
        action: str,
        /,
        **detail: Any,
    ) -> Record:
        """Return a copy with one event appended. Never mutates."""
        event = ProvenanceEvent(stage=stage, action=action, detail=dict(detail))
        return replace(self, provenance=(*self.provenance, event))

    def dropped(self, stage: str, reason: str, /, **detail: Any) -> Record:
        """Mark the record dropped by ``stage`` for ``reason``.

        Dropping is idempotent in intent but not silent: re-dropping an
        already-dropped record is a bug in the caller's stage ordering, so it
        raises rather than overwriting the first, true cause.
        """
        if self.dropped_by is not None:
            raise ValueError(
                f"record {self.record_id} was already dropped by {self.dropped_by!r}; "
                f"{stage!r} should not have seen it"
            )
        return replace(
            self.with_event(stage, f"drop:{reason}", **detail),
            dropped_by=stage,
        )

    def kept(self, stage: str, /, **detail: Any) -> Record:
        """Record that a stage inspected this record and let it through."""
        return self.with_event(stage, "keep", **detail)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Record:
        raw = json.loads(line)
        events = tuple(ProvenanceEvent(**e) for e in raw.pop("provenance", ()))
        return cls(provenance=events, **raw)


def content_id(text: str) -> str:
    """A stable, content-addressed id.

    SHA-256 truncated to 16 hex chars. Truncation is safe at the corpus sizes
    this pipeline targets: the birthday bound puts a collision at roughly
    2**32 records, several orders of magnitude beyond anything here, and the
    shorter id keeps JSONL readable by eye during debugging.

    Content-addressed rather than sequential so that the same generated text
    gets the same id on every run, on every machine. That is what makes the
    exact-duplicate stage trivially correct and the drift gate hold.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class StageCount:
    """One row of the survival funnel."""

    stage: str
    entered: int
    dropped: int

    @property
    def survived(self) -> int:
        return self.entered - self.dropped


def survival_funnel(records: list[Record]) -> list[StageCount]:
    """Reconstruct the funnel from the records themselves.

    Deliberately derived from provenance rather than from counters incremented
    as the pipeline runs. A counter can drift from the data it claims to
    describe; a value recomputed from the surviving artifacts cannot. If this
    function and a hand-kept tally ever disagreed, this one would be right.
    """
    counts: list[StageCount] = []
    live = list(records)
    for stage in STAGES:
        if stage == STAGE_GENERATE:
            continue
        entered = len(live)
        dropped = sum(1 for r in live if r.dropped_by == stage)
        counts.append(StageCount(stage=stage, entered=entered, dropped=dropped))
        live = [r for r in live if r.dropped_by != stage]
    return counts


def drop_reasons(records: list[Record], stage: str) -> dict[str, int]:
    """Why records died at ``stage``, most common first.

    A funnel that says "we dropped 300 rows" without saying why is a number,
    not a finding.
    """
    reasons: dict[str, int] = {}
    for r in records:
        if r.dropped_by != stage:
            continue
        for event in reversed(r.provenance):
            if event.stage == stage and event.action.startswith("drop:"):
                reason = event.action.removeprefix("drop:")
                reasons[reason] = reasons.get(reason, 0) + 1
                break
    return dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
