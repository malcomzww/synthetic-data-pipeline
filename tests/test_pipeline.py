"""The whole pipeline, and the properties that must hold across all of it."""

from __future__ import annotations

from synthetic_data_pipeline.decontaminate import (
    DEFAULT_N,
    EvalIndex,
    decontaminate,
    plant_contamination,
)
from synthetic_data_pipeline.dedup import (
    DEFAULT_THRESHOLD,
    drop_exact_duplicates,
    drop_near_duplicates,
    find_near_duplicates,
)
from synthetic_data_pipeline.eval_set import EVAL_SET
from synthetic_data_pipeline.filters import apply_filters, plant_pii, scrub_pii
from synthetic_data_pipeline.generate import TemplateGenerator, plant_near_duplicates
from synthetic_data_pipeline.lineage import (
    STAGES,
    Record,
    drop_reasons,
    survival_funnel,
)

SEED = 7


def run_pipeline(n: int = 400) -> list[Record]:
    records = TemplateGenerator(seed=SEED).generate(n)
    records, _ = plant_near_duplicates(records, n_pairs=n // 20, seed=SEED)
    records, contam = plant_contamination(records, EVAL_SET, rate=0.12, seed=SEED)
    records, _ = plant_pii(records, rate=0.08, seed=SEED)

    records = apply_filters(records)
    records = drop_exact_duplicates(records)
    records, _ = drop_near_duplicates(records, threshold=DEFAULT_THRESHOLD)
    records, _ = decontaminate(records, EvalIndex.build(EVAL_SET, n=DEFAULT_N), contam)
    return records


class TestFunnelIntegrity:
    def test_the_funnel_balances(self):
        records = run_pipeline()
        funnel = survival_funnel(records)
        survivors = sum(1 for r in records if r.alive)

        assert len(records) - sum(c.dropped for c in funnel) == survivors

    def test_no_record_is_lost_or_duplicated(self):
        # Every stage returns a record for every record it was given, so the
        # dropped ones stay auditable rather than vanishing.
        records = TemplateGenerator(seed=SEED).generate(200)
        after = apply_filters(records)
        after = drop_exact_duplicates(after)
        after, _ = drop_near_duplicates(after)

        assert len(after) == len(records)
        assert {r.record_id for r in after} == {r.record_id for r in records}

    def test_every_dropped_record_says_which_stage_dropped_it(self):
        for record in run_pipeline():
            if not record.alive:
                assert record.dropped_by in STAGES

    def test_every_dropped_record_carries_a_reason(self):
        records = run_pipeline()
        for stage in STAGES:
            dropped = sum(1 for r in records if r.dropped_by == stage)
            assert dropped == sum(drop_reasons(records, stage).values())

    def test_survivors_carry_a_keep_event_from_every_stage_they_passed(self):
        for record in run_pipeline():
            if not record.alive:
                continue
            stages = {e.stage for e in record.provenance if e.action == "keep"}
            assert {"filter", "dedup", "decontaminate"} <= stages


class TestOutputQuality:
    def test_no_survivor_contains_detectable_pii(self):
        assert not any(scrub_pii(r.text).found for r in run_pipeline() if r.alive)

    def test_no_survivor_overlaps_the_eval_set(self):
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        assert not any(index.is_contaminated(r.text) for r in run_pipeline() if r.alive)

    def test_no_survivor_is_an_exact_duplicate_of_another(self):
        from synthetic_data_pipeline.dedup import normalise

        survivors = [r for r in run_pipeline() if r.alive]
        texts = [normalise(r.text) for r in survivors]
        assert len(set(texts)) == len(texts)

    def test_no_survivor_pair_is_above_the_near_duplicate_threshold(self):
        # The stage claims to have removed these; it must actually have done so.
        survivors = [r for r in run_pipeline() if r.alive]
        assert find_near_duplicates(survivors, threshold=DEFAULT_THRESHOLD) == set()

    def test_some_records_survive(self):
        # A pipeline that drops everything would pass every test above.
        assert sum(1 for r in run_pipeline() if r.alive) > 0


class TestDeterminism:
    def test_the_whole_pipeline_is_reproducible(self):
        # The drift gate depends on this: a byte-identical result on every run.
        first = [r.to_json() for r in run_pipeline(200)]
        second = [r.to_json() for r in run_pipeline(200)]
        assert first == second

    def test_provenance_survives_a_jsonl_round_trip(self):
        for record in run_pipeline(100):
            assert Record.from_json(record.to_json()) == record
