"""Provenance must survive every operation the pipeline performs on a record."""

from __future__ import annotations

import pytest

from synthetic_data_pipeline.lineage import (
    STAGE_DECONTAMINATE,
    STAGE_DEDUP,
    STAGE_FILTER,
    STAGE_GENERATE,
    ProvenanceEvent,
    Record,
    content_id,
    drop_reasons,
    survival_funnel,
)


def make(rid: str = "r1", *, text: str = "hello world") -> Record:
    return Record(record_id=rid, instruction=text, response="a response")


def test_events_append_without_mutating_the_original():
    original = make()
    updated = original.with_event(STAGE_FILTER, "keep", tokens=5)

    assert original.provenance == ()
    assert len(updated.provenance) == 1
    assert updated.provenance[0].detail["tokens"] == 5


def test_history_accumulates_in_order():
    record = (
        make()
        .with_event(STAGE_GENERATE, "template")
        .with_event(STAGE_FILTER, "keep")
        .with_event(STAGE_DEDUP, "keep")
    )
    assert [e.stage for e in record.provenance] == [
        STAGE_GENERATE,
        STAGE_FILTER,
        STAGE_DEDUP,
    ]


def test_unknown_stage_is_rejected():
    # A typo must be an error, not a silent new category in the funnel report.
    with pytest.raises(ValueError, match="unknown stage"):
        ProvenanceEvent(stage="dedupe", action="keep")


def test_dropping_marks_the_stage_and_records_the_reason():
    record = make().dropped(STAGE_DEDUP, "exact_duplicate", duplicate_of="r0")

    assert not record.alive
    assert record.dropped_by == STAGE_DEDUP
    assert record.provenance[-1].action == "drop:exact_duplicate"
    assert record.provenance[-1].detail["duplicate_of"] == "r0"


def test_double_drop_raises_rather_than_overwriting_the_first_cause():
    # If a second stage sees an already-dropped record the stage ordering is
    # wrong, and the first cause is the true one. Silently overwriting it
    # would make the funnel attribute the loss to the wrong stage.
    record = make().dropped(STAGE_FILTER, "too_short")
    with pytest.raises(ValueError, match="already dropped"):
        record.dropped(STAGE_DEDUP, "exact_duplicate")


def test_json_round_trip_preserves_provenance():
    record = (
        make()
        .with_event(STAGE_GENERATE, "template", seed=3, model=None)
        .dropped(STAGE_DECONTAMINATE, "ngram_overlap", n=13)
    )
    restored = Record.from_json(record.to_json())

    assert restored == record
    assert restored.dropped_by == STAGE_DECONTAMINATE
    assert restored.provenance[0].detail["seed"] == 3


def test_json_round_trip_survives_non_ascii():
    # The multilingual slice must survive a trip through JSONL on disk.
    record = Record(record_id="r", instruction="请解释一致性哈希环", response="Ответ")
    assert Record.from_json(record.to_json()) == record


def test_content_id_is_stable_and_content_addressed():
    assert content_id("abc") == content_id("abc")
    assert content_id("abc") != content_id("abd")
    assert len(content_id("abc")) == 16


def test_text_joins_both_fields():
    # Dedup and decontamination must agree on what "the content" is.
    record = Record(record_id="r", instruction="ask", response="answer")
    assert record.text == "ask\nanswer"


class TestSurvivalFunnel:
    def test_counts_entries_and_drops_per_stage(self):
        records = [
            make("a").kept(STAGE_FILTER).kept(STAGE_DEDUP).kept(STAGE_DECONTAMINATE),
            make("b").dropped(STAGE_FILTER, "too_short"),
            make("c").kept(STAGE_FILTER).dropped(STAGE_DEDUP, "exact_duplicate"),
            make("d").kept(STAGE_FILTER).kept(STAGE_DEDUP).dropped(
                STAGE_DECONTAMINATE, "ngram_overlap"
            ),
        ]
        funnel = {c.stage: c for c in survival_funnel(records)}

        assert (funnel[STAGE_FILTER].entered, funnel[STAGE_FILTER].dropped) == (4, 1)
        assert (funnel[STAGE_DEDUP].entered, funnel[STAGE_DEDUP].dropped) == (3, 1)
        assert (funnel[STAGE_DECONTAMINATE].entered, funnel[STAGE_DECONTAMINATE].dropped) == (2, 1)

    def test_funnel_balances(self):
        records = [
            make("a").kept(STAGE_FILTER).kept(STAGE_DEDUP).kept(STAGE_DECONTAMINATE),
            make("b").dropped(STAGE_FILTER, "too_short"),
            make("c").kept(STAGE_FILTER).dropped(STAGE_DEDUP, "near_duplicate"),
        ]
        funnel = survival_funnel(records)
        survivors = sum(1 for r in records if r.alive)

        assert len(records) - sum(c.dropped for c in funnel) == survivors

    def test_empty_corpus_produces_zero_rows_not_an_error(self):
        assert all(c.entered == 0 for c in survival_funnel([]))


def test_drop_reasons_are_counted_and_ranked():
    records = [
        make("a").dropped(STAGE_FILTER, "too_short"),
        make("b").dropped(STAGE_FILTER, "too_short"),
        make("c").dropped(STAGE_FILTER, "too_long"),
        make("d").kept(STAGE_FILTER),
    ]
    assert drop_reasons(records, STAGE_FILTER) == {"too_short": 2, "too_long": 1}


def test_drop_reasons_ignores_other_stages():
    records = [make("a").kept(STAGE_FILTER).dropped(STAGE_DEDUP, "exact_duplicate")]
    assert drop_reasons(records, STAGE_FILTER) == {}
    assert drop_reasons(records, STAGE_DEDUP) == {"exact_duplicate": 1}
