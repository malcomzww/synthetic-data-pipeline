"""N-gram decontamination, and the recovery of planted contamination.

The tests that matter here are the ones that pin down *why* the detector
misses what it misses. A recall number without that is not interpretable.
"""

from __future__ import annotations

import pytest

from synthetic_data_pipeline.decontaminate import (
    DEFAULT_N,
    ContaminationReport,
    EvalIndex,
    decontaminate,
    ngrams,
    plant_contamination,
    sweep_n,
    tokenise,
)
from synthetic_data_pipeline.eval_set import EVAL_SET
from synthetic_data_pipeline.generate import TemplateGenerator
from synthetic_data_pipeline.lineage import STAGE_DECONTAMINATE, Record

CLEAN = "Sunlight filtered through the tall grass while the horses grazed by the river."


def rec(rid: str, response: str, instruction: str = "Say something.") -> Record:
    return Record(record_id=rid, instruction=instruction, response=response)


class TestTokenise:
    def test_splits_on_punctuation_and_folds_case(self):
        assert tokenise("Cache, cache!") == ["cache", "cache"]

    def test_punctuation_does_not_change_the_ngram(self):
        # A formatting difference must not let contamination through.
        assert ngrams("a b c, d", n=4) == ngrams("a b c d", n=4)


class TestNgrams:
    def test_produces_overlapping_windows(self):
        assert ngrams("a b c d", n=3) == {"a b c", "b c d"}

    def test_text_shorter_than_n_yields_nothing(self):
        # Correct behaviour, not a gap: a 3-word record cannot contain a
        # 13-gram, and backing off to a shorter n would silently change the
        # threshold the number is reported under.
        assert ngrams("a b c", n=13) == set()

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            ngrams("a b c", n=0)


class TestEvalIndex:
    def test_flags_a_verbatim_span(self):
        index = EvalIndex.build([EVAL_SET[0]], n=DEFAULT_N)
        span = " ".join(tokenise(EVAL_SET[0])[:DEFAULT_N])
        assert index.is_contaminated(f"Some novel text. {span}")

    def test_does_not_flag_unrelated_text(self):
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        assert not index.is_contaminated(CLEAN)

    def test_a_span_shorter_than_n_is_invisible(self):
        # This is the whole limitation of the method, stated as a test.
        index = EvalIndex.build([EVAL_SET[0]], n=DEFAULT_N)
        short = " ".join(tokenise(EVAL_SET[0])[: DEFAULT_N - 1])
        assert not index.is_contaminated(f"Some novel text. {short}")

    def test_overlap_returns_the_matching_grams(self):
        index = EvalIndex.build([EVAL_SET[0]], n=DEFAULT_N)
        span = " ".join(tokenise(EVAL_SET[0])[:DEFAULT_N])
        assert index.overlap(span) == {span}

    def test_records_how_many_items_it_indexed(self):
        assert EvalIndex.build(EVAL_SET, n=DEFAULT_N).n_items == len(EVAL_SET)


class TestDecontaminate:
    def test_drops_contaminated_and_keeps_clean(self):
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        span = " ".join(tokenise(EVAL_SET[0])[:DEFAULT_N])
        records = [rec("dirty", f"Novel text. {span}"), rec("clean", CLEAN)]

        out, report = decontaminate(records, index, {"dirty"})

        assert not out[0].alive and out[0].dropped_by == STAGE_DECONTAMINATE
        assert out[1].alive
        assert report.flagged == 1
        assert report.true_positives == 1

    def test_records_the_matching_ngram_as_evidence(self):
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        span = " ".join(tokenise(EVAL_SET[0])[:DEFAULT_N])
        out, _ = decontaminate([rec("d", f"Novel. {span}")], index, {"d"})

        detail = out[0].provenance[-1].detail
        assert detail["n"] == DEFAULT_N
        assert detail["example"] in EvalIndex.build(EVAL_SET, n=DEFAULT_N).grams

    def test_only_live_records_are_inspected(self):
        # The rate must be over what reached this stage, or a change in the
        # dedup threshold silently moves the headline number.
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        dropped = rec("d", CLEAN).dropped("dedup", "exact_duplicate")
        _, report = decontaminate([dropped, rec("a", CLEAN)], index, set())
        assert report.inspected == 1

    def test_empty_corpus_reports_zero_rather_than_dividing_by_zero(self):
        _, report = decontaminate([], EvalIndex.build(EVAL_SET), set())
        assert report.rate == 0.0 and report.recall == 1.0


class TestPlantContamination:
    def test_plants_the_requested_rate(self):
        base = TemplateGenerator(seed=5).generate(200)
        _, truth = plant_contamination(base, EVAL_SET, rate=0.1, seed=5)
        assert len(truth) == 20

    def test_ids_are_preserved_so_ground_truth_survives_the_splice(self):
        base = TemplateGenerator(seed=5).generate(50)
        out, truth = plant_contamination(base, EVAL_SET, rate=0.2, seed=5)
        assert {r.record_id for r in out} == {r.record_id for r in base}
        assert truth <= {r.record_id for r in base}

    def test_the_splice_is_recorded_in_provenance(self):
        base = TemplateGenerator(seed=5).generate(50)
        out, truth = plant_contamination(base, EVAL_SET, rate=0.2, seed=5)

        for record in out:
            if record.record_id in truth:
                event = record.provenance[-1]
                assert event.detail["planted"] == "contamination"
                assert event.detail["span_tokens"] >= 1

    def test_spans_straddle_the_threshold_in_both_directions(self):
        # If every planted span were >= n the measurement would be
        # tautological: an n-gram detector cannot miss a verbatim n-token
        # span, and recall would read 100% by construction.
        base = TemplateGenerator(seed=5).generate(300)
        out, truth = plant_contamination(base, EVAL_SET, rate=0.3, seed=5)
        lengths = [
            e.detail["span_tokens"]
            for r in out
            if r.record_id in truth
            for e in r.provenance
            if e.detail.get("planted") == "contamination"
        ]
        assert min(lengths) < DEFAULT_N < max(lengths)

    def test_invalid_rate_raises(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="rate must be"):
            plant_contamination(base, EVAL_SET, rate=1.5, seed=0)

    def test_empty_eval_set_raises(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="empty eval set"):
            plant_contamination(base, [], rate=0.1, seed=0)

    def test_eval_texts_too_short_to_supply_a_span_raise(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="cannot plant a span"):
            plant_contamination(
                base, ["too short"], rate=0.1, seed=0, min_span=50, max_span=60
            )

    def test_an_inverted_span_range_raises(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="min_span <= max_span"):
            plant_contamination(base, EVAL_SET, rate=0.1, seed=0, min_span=30, max_span=10)


class TestRecoveryOfPlantedContamination:
    """The repo's headline claim, as an executable test."""

    def _run(self, n_records: int = 400, rate: float = 0.15, seed: int = 7):
        base = TemplateGenerator(seed=seed).generate(n_records)
        records, truth = plant_contamination(base, EVAL_SET, rate=rate, seed=seed)
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)
        _, report = decontaminate(records, index, truth)
        return records, truth, report

    def test_detection_is_decided_entirely_by_span_length(self):
        # The sharp claim: every span of >= n tokens is caught and every
        # shorter span is missed, with no exceptions in either direction.
        records, truth, _ = self._run()
        index = EvalIndex.build(EVAL_SET, n=DEFAULT_N)

        for record in records:
            if record.record_id not in truth:
                continue
            span = next(
                e.detail["span_tokens"]
                for e in record.provenance
                if e.detail.get("planted") == "contamination"
            )
            detected = index.is_contaminated(record.text)
            assert detected == (span >= DEFAULT_N), (
                f"span of {span} tokens was {'caught' if detected else 'missed'}"
            )

    def test_no_clean_record_is_ever_flagged(self):
        _, _, report = self._run()
        assert report.false_positives == 0
        assert report.precision == 1.0

    def test_recall_and_miss_rate_are_complements(self):
        _, _, report = self._run()
        assert report.recall + report.miss_rate == pytest.approx(1.0)

    def test_recall_falls_monotonically_as_n_rises(self):
        base = TemplateGenerator(seed=7).generate(400)
        records, truth = plant_contamination(base, EVAL_SET, rate=0.15, seed=7)

        recalls = [r.recall for r in sweep_n(records, EVAL_SET, truth, (5, 8, 13, 20))]
        assert recalls == sorted(recalls, reverse=True)

    def test_sweep_on_an_already_decontaminated_corpus_raises(self):
        # A sweep over a corpus whose contaminated rows were already dropped
        # reports all-zero, which reads like a finding ("no contamination at
        # any n") but is a caller error.
        base = TemplateGenerator(seed=7).generate(200)
        records, truth = plant_contamination(base, EVAL_SET, rate=0.15, seed=7)
        cleaned, _ = decontaminate(records, EvalIndex.build(EVAL_SET, n=5), truth)

        with pytest.raises(ValueError, match="before decontaminate"):
            sweep_n(cleaned, EVAL_SET, truth, (13,))


class TestContaminationReport:
    def test_rates_are_ratios_of_counts(self):
        report = ContaminationReport(
            n=13, inspected=100, flagged=10, planted=20, true_positives=10,
            false_positives=0,
        )
        assert report.rate == 0.1
        assert report.planted_rate == 0.2
        assert report.recall == 0.5
        assert report.miss_rate == 0.5

    def test_nothing_planted_does_not_divide_by_zero(self):
        report = ContaminationReport(
            n=13, inspected=10, flagged=0, planted=0, true_positives=0, false_positives=0,
        )
        assert report.recall == 1.0
