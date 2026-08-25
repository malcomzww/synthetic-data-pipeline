"""Length, script detection and PII scrubbing."""

from __future__ import annotations

import pytest

from synthetic_data_pipeline.filters import (
    PII_SPECIMENS,
    apply_filters,
    detect_script,
    plant_pii,
    score_pii,
    script_profile,
    scrub_pii,
)
from synthetic_data_pipeline.generate import TemplateGenerator
from synthetic_data_pipeline.lineage import Record


def rec(rid: str, instruction: str, response: str = "a perfectly ordinary response") -> Record:
    return Record(record_id=rid, instruction=instruction, response=response)


class TestPIIPatterns:
    @pytest.mark.parametrize("kind,specimen", PII_SPECIMENS)
    def test_each_specimen_is_detected(self, kind: str, specimen: str):
        assert scrub_pii(specimen).found

    @pytest.mark.parametrize("kind,specimen", PII_SPECIMENS)
    def test_each_specimen_is_labelled_correctly(self, kind: str, specimen: str):
        # Ordering in PII_PATTERNS is load-bearing. A greedy phone pattern
        # once swallowed the SSN, card, IPv4 and IBAN and labelled all four
        # "phone" -- every specimen was still detected, so a detected-or-not
        # test stayed green while the audit log was wrong.
        assert kind in scrub_pii(specimen).kinds

    @pytest.mark.parametrize("kind,specimen", PII_SPECIMENS)
    def test_the_identifier_is_removed_from_the_text(self, kind: str, specimen: str):
        scrubbed = scrub_pii(specimen).text
        assert f"[REDACTED:{kind}]" in scrubbed

    def test_redaction_is_typed_not_blank(self):
        # A typed marker preserves the sentence shape the model learns from.
        out = scrub_pii("Mail ada@example.org now.").text
        assert out == "Mail [REDACTED:email] now."

    def test_clean_prose_is_left_alone(self):
        clean = "The cache holds four thousand entries and evicts the oldest."
        result = scrub_pii(clean)
        assert not result.found
        assert result.text == clean

    def test_a_year_range_is_not_mistaken_for_a_phone_number(self):
        assert not scrub_pii("Between 1990 and 2020 the design changed.").found

    def test_a_version_string_is_not_mistaken_for_an_address(self):
        assert not scrub_pii("Upgrade to version 3.11 before Friday.").found


class TestScriptDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hello world", "latin"),
            ("журнал предзаписи", "cyrillic"),
            ("ουρά προτεραιότητας", "greek"),
            ("व्युत्क्रम सूचकांक", "devanagari"),
            ("مخزن الأعمدة", "arabic"),
            ("一致性哈希环", "han"),
            ("우선순위 큐", "hangul"),
        ],
    )
    def test_identifies_the_dominant_script(self, text: str, expected: str):
        assert detect_script(text).dominant == expected

    def test_cannot_distinguish_languages_sharing_a_script(self):
        # The boundary of the claim, asserted so it cannot quietly widen:
        # this is script detection, not language identification.
        spanish = detect_script("una cola de prioridad ordena los elementos")
        portuguese = detect_script("uma fila de prioridade ordena os elementos")
        assert spanish.dominant == portuguese.dominant == "latin"

    def test_detects_code_switching(self):
        verdict = detect_script("बताइए कि rate limiter को production में deploy करते हैं")
        assert verdict.code_switched
        assert len(verdict.profile) > 1

    def test_incidental_latin_does_not_count_as_code_switching(self):
        # Latin leaks into every script through product names and units.
        # Flagging that would mark most of the multilingual slice.
        verdict = detect_script("一致性哈希环把键和节点映射到同一个环上从而减少重新分配ok")
        assert not verdict.code_switched

    def test_text_without_letters_is_unknown(self):
        assert detect_script("12345 !!!").dominant == "unknown"

    def test_profile_sums_to_one(self):
        profile = script_profile("hello мир")
        assert sum(profile.values()) == pytest.approx(1.0)


class TestApplyFilters:
    def test_drops_text_below_the_length_floor(self):
        out = apply_filters([rec("a", "hi", response="yo")], min_tokens=6)
        assert not out[0].alive
        assert out[0].provenance[-1].action == "drop:too_short"

    def test_drops_text_above_the_length_ceiling(self):
        out = apply_filters([rec("a", "word " * 50)], max_tokens=10)
        assert out[0].provenance[-1].action == "drop:too_long"

    def test_keeps_ordinary_records_and_records_why(self):
        out = apply_filters([rec("a", "Explain how a bloom filter works in practice")])
        assert out[0].alive
        detail = out[0].provenance[-1].detail
        assert detail["script"] == "latin"
        assert detail["tokens"] > 0

    def test_pii_is_scrubbed_rather_than_dropped(self):
        # Dropping every record containing an email would bias the corpus
        # away from anything about contacting people.
        record = rec("a", "Write to ada@example.org about the outage please")
        out = apply_filters([record])

        assert out[0].alive
        assert "ada@example.org" not in out[0].text
        assert "email" in out[0].provenance[-1].detail["pii_scrubbed"]

    def test_scrubbing_can_be_disabled(self):
        record = rec("a", "Write to ada@example.org about the outage please")
        out = apply_filters([record], scrub=False)
        assert "ada@example.org" in out[0].text

    def test_disallowed_scripts_are_dropped_when_a_whitelist_is_given(self):
        out = apply_filters(
            [rec("a", "一致性哈希环把键和节点映射到同一个环上从而减少重新分配")],
            allowed_scripts=frozenset({"latin"}),
            min_tokens=1,
        )
        assert out[0].provenance[-1].action == "drop:script_not_allowed"

    def test_all_scripts_pass_by_default(self):
        out = apply_filters([rec("a", "журнал предзаписи сохраняет изменения до применения")])
        assert out[0].alive

    def test_already_dropped_records_pass_through(self):
        dropped = rec("a", "x").dropped("generate", "synthetic")
        assert apply_filters([dropped])[0].dropped_by == "generate"


class TestRecoveryOfPlantedPII:
    def test_every_planted_identifier_is_found(self):
        base = TemplateGenerator(seed=7).generate(300)
        records, truth = plant_pii(base, rate=0.2, seed=7)
        score = score_pii(records, truth)

        assert score.recall == 1.0, f"missed {score.false_negatives} planted identifiers"

    def test_no_clean_record_is_flagged(self):
        base = TemplateGenerator(seed=7).generate(300)
        records, truth = plant_pii(base, rate=0.2, seed=7)
        score = score_pii(records, truth)

        assert score.false_positives == 0
        assert score.precision == 1.0

    def test_plants_the_requested_rate(self):
        base = TemplateGenerator(seed=3).generate(200)
        _, truth = plant_pii(base, rate=0.1, seed=3)
        assert len(truth) == 20

    def test_ids_are_preserved_so_ground_truth_survives(self):
        base = TemplateGenerator(seed=3).generate(100)
        out, _ = plant_pii(base, rate=0.2, seed=3)
        assert {r.record_id for r in out} == {r.record_id for r in base}

    def test_the_splice_is_recorded_in_provenance(self):
        base = TemplateGenerator(seed=3).generate(50)
        out, truth = plant_pii(base, rate=0.2, seed=3)
        for record in out:
            if record.record_id in truth:
                assert record.provenance[-1].detail["planted"] == "pii"

    def test_filtering_removes_every_planted_identifier(self):
        # End to end: after the filter stage nothing PII-shaped remains.
        base = TemplateGenerator(seed=7).generate(200)
        records, _ = plant_pii(base, rate=0.25, seed=7)
        filtered = apply_filters(records)

        assert not any(scrub_pii(r.text).found for r in filtered if r.alive)

    def test_invalid_rate_raises(self):
        base = TemplateGenerator(seed=0).generate(10)
        with pytest.raises(ValueError, match="rate must be"):
            plant_pii(base, rate=-0.1, seed=0)


def test_score_pii_handles_an_empty_ground_truth():
    assert score_pii([], set()).recall == 1.0
