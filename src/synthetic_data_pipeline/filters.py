"""Quality gates: length, language ID, PII scrubbing.

Ordered cheapest-first. Length is a comparison, script detection is a linear
scan, PII is a set of regexes -- so the expensive one only ever sees what the
cheap ones let through.

**On language ID.** The brief's stack names fastText. This module implements
script-based detection instead, and the substitution is deliberate rather than
a shortcut: fastText's `lid.176.bin` is a 126 MB model download, which makes a
fresh clone fail on a machine without network access and makes CI depend on a
third-party file server. What script detection actually delivers is narrower
and is labelled as such -- it separates Latin from Cyrillic from Han, and it
detects code-switching by finding more than one script in one record. It
**cannot** tell Spanish from Portuguese, because they share a script. Any claim
this repo makes about language is a claim about *script*, and the tests assert
exactly that boundary and no more.

**On PII.** Regex PII detection is a recall-oriented filter, not a guarantee.
It catches the formatted identifiers -- emails, phone numbers, card-shaped
digit runs -- and is structurally blind to unformatted PII: a person's name, a
home address in prose, a date of birth written as words. The measured
precision/recall below is against *planted* patterns, so it measures whether
the regexes fire on the shapes they were written for. It is not evidence that
a corpus is PII-free, and this repo does not claim that.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from .lineage import STAGE_FILTER, Record

MIN_TOKENS = 4
MAX_TOKENS = 512

# --- PII patterns -----------------------------------------------------
#
# Each is deliberately anchored on structure that plain prose does not have.
# The looser a pattern, the more real content it eats: an unanchored 9-digit
# run matches order numbers, checksums, and timestamps.
#
# **Order is load-bearing, most specific first.** The first version of this
# tuple listed `phone` second, and its digit-group pattern then swallowed the
# SSN, the card number, the IPv4 address and half the IBAN, redacting all four
# as "[REDACTED:phone]". Every specimen was still caught, so a
# detected-or-not test stayed green while the *labels* were wrong -- and a
# scrubber that mislabels what it removed produces an audit log that cannot be
# trusted. The specific patterns now run first and consume their matches, so
# `phone` only ever sees what the others left behind.

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    ),
    (
        "ipv4",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?\d{1,2})?\b"),
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        # 13-19 digits, optionally grouped by space or hyphen. Deliberately
        # not Luhn-validated: a scrubber that only redacts *valid* card
        # numbers leaks every typo of a real one, and the point is to remove
        # the shape rather than to confirm the account exists.
        "credit_card",
        re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{1,7}(?!\d)|(?<!\d)\d{13,19}(?!\d)"),
    ),
    (
        # Runs last, so the patterns above have already consumed the shapes it
        # would otherwise mislabel. Requires either a leading + or a
        # parenthesised area code -- a bare run of digit groups is far more
        # often a version string or a date than a telephone number.
        "phone",
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){1,4}"
            r"|\(\d{2,4}\)[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4}){1,3})(?!\w)"
        ),
    ),
)

REDACTION = "[REDACTED:{kind}]"


@dataclass(frozen=True, slots=True)
class ScrubResult:
    text: str
    kinds: tuple[str, ...]

    @property
    def found(self) -> bool:
        return bool(self.kinds)


def scrub_pii(text: str) -> ScrubResult:
    """Replace recognised PII with a typed redaction marker.

    A *typed* marker rather than a blank, because "[REDACTED:email]" preserves
    the sentence structure the model is learning from while removing the
    identifier. Deleting the span outright teaches the model that sentences
    sometimes just stop.

    Patterns are applied in declaration order, and each match is replaced
    before the next pattern runs, so an email is not subsequently re-matched as
    a phone number by the digits in its domain.
    """
    kinds: list[str] = []
    out = text
    for kind, pattern in PII_PATTERNS:
        if pattern.search(out):
            kinds.append(kind)
            out = pattern.sub(REDACTION.format(kind=kind), out)
    return ScrubResult(text=out, kinds=tuple(kinds))


# --- script / language ID ---------------------------------------------

_SCRIPT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("LATIN", "latin"),
    ("CYRILLIC", "cyrillic"),
    ("GREEK", "greek"),
    ("ARABIC", "arabic"),
    ("HEBREW", "hebrew"),
    ("DEVANAGARI", "devanagari"),
    ("CJK", "han"),
    ("HIRAGANA", "kana"),
    ("KATAKANA", "kana"),
    ("HANGUL", "hangul"),
    ("THAI", "thai"),
)


def _char_script(ch: str) -> str | None:
    """Map one character to a coarse script name via its Unicode name.

    Using the character's Unicode name rather than a hand-maintained codepoint
    range table: the ranges are long, easy to get subtly wrong, and go stale
    with each Unicode release, while the names are authoritative and shipped
    with Python.
    """
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for prefix, script in _SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return script
    return "other"


def script_profile(text: str) -> dict[str, float]:
    """Fraction of alphabetic characters in each script. Sums to 1.0 or is empty."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        script = _char_script(ch)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if not total:
        return {}
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


@dataclass(frozen=True, slots=True)
class ScriptVerdict:
    dominant: str
    profile: dict[str, float]
    code_switched: bool


def detect_script(text: str, *, secondary_threshold: float = 0.15) -> ScriptVerdict:
    """Dominant script, plus whether a second script holds a real share.

    ``secondary_threshold`` at 0.15 rather than something tiny: Latin
    characters leak into every script through product names, units, and code
    identifiers, and calling every such record code-switched would flag most of
    the multilingual slice for a property it does not have.
    """
    profile = script_profile(text)
    if not profile:
        return ScriptVerdict(dominant="unknown", profile={}, code_switched=False)
    ranked = list(profile.items())
    dominant = ranked[0][0]
    switched = len(ranked) > 1 and ranked[1][1] >= secondary_threshold
    return ScriptVerdict(dominant=dominant, profile=profile, code_switched=switched)


# --- the stage --------------------------------------------------------


def token_count(text: str) -> int:
    return len(text.split())


def apply_filters(
    records: Sequence[Record],
    *,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    allowed_scripts: frozenset[str] | None = None,
    scrub: bool = True,
) -> list[Record]:
    """Run the filter stage. Scrubbing rewrites; length and script drop.

    PII **scrubs rather than drops** -- that is the substantive choice in this
    module. Dropping every record containing an email address would remove a
    large slice of otherwise good instruction data and bias the corpus away
    from anything that mentions contacting someone. Redaction keeps the
    pedagogical content and removes the identifier.

    ``allowed_scripts=None`` means every script passes; the parameter exists so
    a caller building a single-script corpus can say so explicitly rather than
    discovering the mix later.
    """
    out: list[Record] = []
    for record in records:
        if not record.alive:
            out.append(record)
            continue

        current = record
        n_tokens = token_count(current.text)
        if n_tokens < min_tokens:
            out.append(current.dropped(STAGE_FILTER, "too_short", tokens=n_tokens))
            continue
        if n_tokens > max_tokens:
            out.append(current.dropped(STAGE_FILTER, "too_long", tokens=n_tokens))
            continue

        verdict = detect_script(current.text)
        if allowed_scripts is not None and verdict.dominant not in allowed_scripts:
            out.append(
                current.dropped(STAGE_FILTER, "script_not_allowed", script=verdict.dominant)
            )
            continue

        detail: dict[str, object] = {
            "tokens": n_tokens,
            "script": verdict.dominant,
            "code_switched": verdict.code_switched,
        }

        if scrub:
            si = scrub_pii(current.instruction)
            sr = scrub_pii(current.response)
            kinds = tuple(dict.fromkeys(si.kinds + sr.kinds))
            if kinds:
                current = current.__class__(
                    record_id=current.record_id,
                    instruction=si.text,
                    response=sr.text,
                    language=current.language,
                    provenance=current.provenance,
                    dropped_by=current.dropped_by,
                )
                detail["pii_scrubbed"] = list(kinds)

        out.append(current.kept(STAGE_FILTER, **detail))
    return out


@dataclass(frozen=True, slots=True)
class PIIScore:
    """Detector performance against planted PII, per record."""

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


def score_pii(records: Sequence[Record], planted: set[str]) -> PIIScore:
    """Score the scrubber against the ids PII was planted into.

    Per record, not per span: a record whose email was caught but whose phone
    number was missed still counts as a hit here. That makes this an optimistic
    metric and it is named as such -- the span-level number would be lower.
    """
    tp = fp = fn = 0
    for record in records:
        found = scrub_pii(record.text).found
        planted_here = record.record_id in planted
        if found and planted_here:
            tp += 1
        elif found and not planted_here:
            fp += 1
        elif not found and planted_here:
            fn += 1
    return PIIScore(true_positives=tp, false_positives=fp, false_negatives=fn)


PII_SPECIMENS: tuple[tuple[str, str], ...] = (
    ("email", "Contact the maintainer at ada.lovelace@example.org for access."),
    ("phone", "Escalation line: +44 20 7946 0958 during business hours."),
    ("ssn", "The legacy record was keyed on 123-45-6789 which must not ship."),
    ("credit_card", "Test card 4111 1111 1111 1111 was left in the fixture."),
    ("ipv4", "The failing node was 192.168.13.240 in the staging rack."),
    ("iban", "Settlement account GB29 NWBK 6016 1331 9268 19 appeared in a log."),
)


def plant_pii(
    records: Sequence[Record],
    *,
    rate: float,
    seed: int = 0,
) -> tuple[list[Record], set[str]]:
    """Splice a PII specimen into a chosen fraction of the corpus.

    Returns the modified corpus and the ground-truth ids. Specimens are
    documentation-reserved values (RFC 2606 domains, RFC 1918 addresses, the
    published Visa test number, the IBAN from the UK banking spec) so no real
    identifier is committed to this repository to test a redactor.

    Ids are not recomputed, for the same reason as in `plant_contamination`:
    ground truth is keyed on them.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0,1], got {rate}")

    import random

    rng = random.Random(seed)
    target = int(round(len(records) * rate))
    chosen = set(rng.sample(range(len(records)), min(target, len(records))))

    out: list[Record] = []
    truth: set[str] = set()
    for i, record in enumerate(records):
        if i not in chosen:
            out.append(record)
            continue
        _, specimen = rng.choice(PII_SPECIMENS)
        out.append(
            record.__class__(
                record_id=record.record_id,
                instruction=record.instruction,
                response=record.response + " " + specimen,
                language=record.language,
                provenance=record.provenance,
                dropped_by=record.dropped_by,
            ).with_event("generate", "splice", planted="pii")
        )
        truth.add(record.record_id)
    return out, truth
