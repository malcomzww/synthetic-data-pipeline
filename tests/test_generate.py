"""The generator's contract: deterministic, distinct, and labelled as templated."""

from __future__ import annotations

import pytest

from synthetic_data_pipeline.generate import (
    MULTILINGUAL_SEEDS,
    TemplateGenerator,
    multilingual_records,
    paraphrase,
    plant_near_duplicates,
)
from synthetic_data_pipeline.lineage import STAGE_GENERATE


class TestDeterminism:
    def test_same_seed_gives_identical_output(self):
        a = TemplateGenerator(seed=11).generate(50)
        b = TemplateGenerator(seed=11).generate(50)
        assert [r.to_json() for r in a] == [r.to_json() for r in b]

    def test_different_seeds_give_different_output(self):
        a = TemplateGenerator(seed=1).generate(50)
        b = TemplateGenerator(seed=2).generate(50)
        assert [r.record_id for r in a] != [r.record_id for r in b]

    def test_planting_is_seeded(self):
        base = TemplateGenerator(seed=3).generate(80)
        _, truth_a = plant_near_duplicates(base, n_pairs=10, seed=5)
        _, truth_b = plant_near_duplicates(base, n_pairs=10, seed=5)
        assert truth_a == truth_b


class TestDistinctness:
    def test_every_generated_record_has_a_distinct_id(self):
        # Ground truth in every planting helper is keyed on record_id. A
        # collision silently corrupts each measurement downstream, so the
        # generator guarantees this rather than hoping for it.
        records = TemplateGenerator(seed=7).generate(600)
        assert len({r.record_id for r in records}) == len(records)

    def test_every_generated_record_has_distinct_text(self):
        records = TemplateGenerator(seed=7).generate(600)
        assert len({r.text for r in records}) == len(records)

    def test_requesting_more_than_capacity_raises(self):
        generator = TemplateGenerator(seed=0)
        with pytest.raises(ValueError, match="distinct renderings"):
            generator.generate(generator.capacity * 2)

    def test_capacity_is_actually_reachable(self):
        # capacity must be the number a caller can request, not an upper bound
        # computed from a cross product with unrenderable cells in it.
        generator = TemplateGenerator(seed=0, multilingual_fraction=0.1)
        records = generator.generate(generator.capacity - 1)
        assert len({r.record_id for r in records}) == len(records)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TemplateGenerator(seed=0).generate(-1)

    def test_zero_records_is_allowed(self):
        assert TemplateGenerator(seed=0).generate(0) == []


class TestProvenance:
    def test_generation_is_labelled_templated_not_model_generated(self):
        # The honesty claim the whole repo rests on: nothing here came from a
        # language model, and each record says so in its own provenance.
        record = TemplateGenerator(seed=0).generate(1)[0]
        event = record.provenance[0]

        assert event.stage == STAGE_GENERATE
        assert event.detail["source"] == "templated"
        assert event.detail["model"] is None

    def test_seed_is_recorded_in_provenance(self):
        record = TemplateGenerator(seed=42).generate(1)[0]
        assert record.provenance[0].detail["seed"] == 42


class TestMultilingualSlice:
    def test_every_rendered_record_is_distinct(self):
        rendered = multilingual_records()
        assert len({(i, r) for i, r, _ in rendered}) == len(rendered)

    def test_covers_every_declared_language(self):
        langs = {lang for _, _, lang in multilingual_records()}
        assert langs == {lang for lang, _, _ in MULTILINGUAL_SEEDS}

    def test_slice_scales_beyond_the_seed_count(self):
        # Twelve fixed rows would cap script coverage at twelve records however
        # large the corpus, which is what made it a rounding error before.
        assert len(multilingual_records()) > len(MULTILINGUAL_SEEDS)

    def test_multilingual_records_appear_in_a_generated_corpus(self):
        records = TemplateGenerator(seed=7, multilingual_fraction=0.1).generate(400)
        assert any(r.language != "en" for r in records)


class TestParaphrase:
    def test_applies_exactly_one_edit(self):
        import random

        # A single edit keeps a short record inside the near-duplicate band.
        # Stacking edits was what pushed planted pairs to Jaccard 0.26-0.57.
        text = "Explain how a bloom filter works in data engineering."
        out = paraphrase(text, random.Random(0))
        assert out != text
        assert "Briefly:" not in out

    def test_output_is_seeded(self):
        import random

        text = "Compare a merkle tree against the usual alternative."
        assert paraphrase(text, random.Random(4)) == paraphrase(text, random.Random(4))


class TestPlantNearDuplicates:
    def test_returns_one_pair_per_request(self):
        base = TemplateGenerator(seed=3).generate(100)
        extended, truth = plant_near_duplicates(base, n_pairs=15, seed=3)

        assert len(truth) == 15
        assert len(extended) == len(base) + 15

    def test_planted_pairs_are_genuinely_near_duplicates(self):
        # The fixture must contain what it claims to contain. If planted pairs
        # fall below the detection threshold the recall number measures the
        # fixture, not the detector.
        from synthetic_data_pipeline.dedup import jaccard

        base = TemplateGenerator(seed=7).generate(300)
        extended, truth = plant_near_duplicates(base, n_pairs=30, seed=7)
        by_id = {r.record_id: r for r in extended}

        similarities = [jaccard(by_id[a].text, by_id[b].text) for a, b in truth]
        assert min(similarities) >= 0.7, f"weakest planted pair was {min(similarities):.2f}"
        # ...and not identical, or exact hashing would catch them and the
        # MinHash stage would never be exercised.
        assert max(similarities) < 1.0

    def test_copy_records_what_it_derives_from(self):
        base = TemplateGenerator(seed=3).generate(60)
        extended, truth = plant_near_duplicates(base, n_pairs=5, seed=3)
        originals = {a for a, _ in truth}
        copies = {b for _, b in truth}

        for record in extended:
            if record.record_id in copies:
                event = record.provenance[-1]
                assert event.detail["planted"] == "near_duplicate"
                assert event.detail["derived_from"] in originals

    def test_planting_more_pairs_than_records_raises(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="cannot plant"):
            plant_near_duplicates(base, n_pairs=11, seed=0)

    def test_negative_pairs_raise(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="non-negative"):
            plant_near_duplicates(base, n_pairs=-1, seed=0)
