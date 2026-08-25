"""Exact and near-duplicate detection, scored against planted pairs."""

from __future__ import annotations

import pytest

from synthetic_data_pipeline.dedup import (
    drop_exact_duplicates,
    drop_near_duplicates,
    find_near_duplicates,
    jaccard,
    normalise,
    score_pairs,
    shingles,
    sweep_thresholds,
)
from synthetic_data_pipeline.generate import TemplateGenerator, plant_near_duplicates
from synthetic_data_pipeline.lineage import STAGE_DEDUP, Record


def rec(rid: str, instruction: str, response: str = "r") -> Record:
    return Record(record_id=rid, instruction=instruction, response=response)


class TestNormalisation:
    def test_collapses_whitespace_and_case(self):
        assert normalise("  Hello   World  ") == "hello world"

    def test_does_not_strip_punctuation(self):
        # Aggressive normalisation collapses records a human calls distinct.
        assert "," in normalise("a, b")


class TestShingles:
    def test_produces_character_ngrams(self):
        assert shingles("abcdef", size=5) == {"abcde", "bcdef"}

    def test_text_shorter_than_the_window_yields_itself(self):
        assert shingles("abc", size=5) == {"abc"}

    def test_empty_text_yields_nothing(self):
        assert shingles("", size=5) == set()

    def test_works_on_unsegmented_script(self):
        # Character shingles are used precisely because whitespace tokenising
        # degenerates on scripts without word boundaries.
        assert len(shingles("一致性哈希环把键和节点映射", size=3)) > 5

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="shingle size"):
            shingles("abc", size=0)


class TestJaccard:
    def test_identical_text_scores_one(self):
        assert jaccard("hello world", "hello world") == 1.0

    def test_case_and_whitespace_do_not_matter(self):
        assert jaccard("Hello  World", "hello world") == 1.0

    def test_disjoint_text_scores_low(self):
        assert jaccard("aaaaaaaaaa", "bbbbbbbbbb") == 0.0

    def test_two_empty_texts_are_equal_not_undefined(self):
        assert jaccard("", "") == 1.0


class TestExactDuplicates:
    def test_keeps_the_first_and_drops_later_repeats(self):
        records = [rec("a", "same text"), rec("b", "same text"), rec("c", "other")]
        out = drop_exact_duplicates(records)

        assert out[0].alive and out[2].alive
        assert not out[1].alive
        assert out[1].dropped_by == STAGE_DEDUP
        assert out[1].provenance[-1].detail["duplicate_of"] == "a"

    def test_matches_across_case_and_whitespace(self):
        records = [rec("a", "Same  Text"), rec("b", "same text")]
        assert not drop_exact_duplicates(records)[1].alive

    def test_already_dropped_records_are_passed_through_untouched(self):
        # Re-dropping would raise; the stage must skip what it did not admit.
        dropped = rec("a", "x").dropped("filter", "too_short")
        out = drop_exact_duplicates([dropped, rec("b", "y")])
        assert out[0].dropped_by == "filter"


class TestNearDuplicates:
    def test_finds_a_paraphrased_pair(self):
        a = rec("a", "Explain how a bloom filter works in data engineering today")
        b = rec("b", "Explain how a bloom filter works in data engineering now")
        assert find_near_duplicates([a, b], threshold=0.7) == {("a", "b")}

    def test_ignores_unrelated_records(self):
        a = rec("a", "Explain how a bloom filter works")
        b = rec("b", "Describe the migration of arctic terns across the pacific")
        assert find_near_duplicates([a, b], threshold=0.7) == set()

    def test_candidates_are_verified_against_true_jaccard(self):
        # LSH returns candidates by banding, which admits pairs below the
        # threshold. Without verification precision drifts below what the
        # threshold implies.
        a = rec("a", "the quick brown fox jumps over the lazy dog repeatedly")
        b = rec("b", "a completely different sentence about something else now")
        for pair in find_near_duplicates([a, b], threshold=0.9):
            assert jaccard(a.text, b.text) >= 0.9, pair

    def test_pairs_are_canonically_ordered(self):
        a = rec("zzz", "Explain how a bloom filter works in data engineering today")
        b = rec("aaa", "Explain how a bloom filter works in data engineering now")
        assert find_near_duplicates([a, b], threshold=0.7) == {("aaa", "zzz")}

    def test_dropped_records_do_not_participate(self):
        # A record already removed as an exact duplicate must not also be
        # counted as a near duplicate, or the two stages double-count.
        a = rec("a", "Explain how a bloom filter works in data engineering today")
        b = rec("b", "Explain how a bloom filter works in data engineering now").dropped(
            STAGE_DEDUP, "exact_duplicate"
        )
        assert find_near_duplicates([a, b], threshold=0.7) == set()

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            find_near_duplicates([rec("a", "x")], threshold=0.0)

    def test_drop_keeps_the_earlier_record(self):
        a = rec("a", "Explain how a bloom filter works in data engineering today")
        b = rec("b", "Explain how a bloom filter works in data engineering now")
        out, _ = drop_near_duplicates([a, b], threshold=0.7)

        assert out[0].alive
        assert not out[1].alive
        assert out[1].provenance[-1].detail["duplicate_of"] == "a"


class TestRecoveryOfPlantedDuplicates:
    """The measurable claim: planted pairs are recovered at a known rate."""

    def test_recall_is_high_at_the_default_threshold(self):
        base = TemplateGenerator(seed=7).generate(300)
        records, truth = plant_near_duplicates(base, n_pairs=30, seed=7)
        records = drop_exact_duplicates(records)

        detected = find_near_duplicates(records, threshold=0.7)
        score = score_pairs(detected, truth, threshold=0.7)

        assert score.recall >= 0.9, f"recovered only {score.recall:.0%} of planted pairs"

    def test_recall_falls_monotonically_as_the_threshold_rises(self):
        base = TemplateGenerator(seed=7).generate(300)
        records, truth = plant_near_duplicates(base, n_pairs=30, seed=7)
        records = drop_exact_duplicates(records)

        recalls = [s.recall for s in sweep_thresholds(records, truth, (0.5, 0.7, 0.9))]
        assert recalls == sorted(recalls, reverse=True)

    def test_planted_pairs_survive_exact_dedup(self):
        # If paraphrasing ever became a no-op the pairs would be caught by
        # exact hashing and the MinHash stage would never be tested.
        base = TemplateGenerator(seed=7).generate(200)
        records, truth = plant_near_duplicates(base, n_pairs=20, seed=7)
        after = {r.record_id for r in drop_exact_duplicates(records) if r.alive}

        for original, copy in truth:
            assert original in after and copy in after


class TestScoring:
    def test_counts_are_computed_against_ground_truth(self):
        score = score_pairs({("a", "b"), ("c", "d")}, {("a", "b"), ("e", "f")})

        assert score.true_positives == 1
        assert score.false_positives == 1
        assert score.false_negatives == 1
        assert score.recall == 0.5

    def test_pair_order_does_not_affect_scoring(self):
        assert score_pairs({("b", "a")}, {("a", "b")}).true_positives == 1

    def test_perfect_detection_scores_one(self):
        score = score_pairs({("a", "b")}, {("a", "b")})
        assert score.precision == score.recall == score.f1 == 1.0

    def test_empty_ground_truth_does_not_divide_by_zero(self):
        assert score_pairs(set(), set()).recall == 1.0
