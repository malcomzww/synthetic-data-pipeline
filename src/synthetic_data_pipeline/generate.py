"""Templated synthetic generation -- the stand-in for the LLM step.

**These records are templated, not model-generated.** No language model was
called anywhere in this repo. That is a deliberate scope decision, not a
missing piece:

The deliverable here is the *hygiene layer*. Dedup, decontamination, PII
scrubbing and language ID are the parts that carry engineering risk, and every
one of them is measurable only against ground truth you control. A real LLM
generator would produce a corpus whose true duplicate count and true
contamination rate are unknown, which means the detectors could be scored
only against each other -- circular, and worth nothing. A templated generator
with planted, counted duplicates gives exact ground truth, so detector recall
becomes a real measurement rather than an impression.

The seam is kept explicit. `Generator` is a protocol; `TemplateGenerator`
implements it; swapping in an LLM-backed generator means implementing the same
three-line interface and changing one line in the pipeline. What it would
*not* do is make the hygiene numbers below reproducible, so the templated path
stays the one that is measured.

Determinism: everything derives from a single seed. The same seed produces
byte-identical output on any machine, which is what lets the results file be
committed and diff-gated.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from .lineage import STAGE_GENERATE, Record, content_id

# --- the template inventory -------------------------------------------
#
# Recombinant by construction: a task frame crossed with a domain crossed with
# an object. The cross product is large enough that near-duplicates arise from
# planting rather than by accident, which keeps the ground truth clean.

TASK_FRAMES: tuple[tuple[str, str], ...] = (
    ("Explain how {obj} works in {domain}.",
     "In {domain}, {obj} works by breaking the problem into stages and handling each in turn."),
    ("Write a short summary of {obj} for a {domain} audience.",
     "For a {domain} audience, {obj} is best summarised as a trade between cost and accuracy."),
    ("What are the main trade-offs of {obj} in {domain}?",
     "The main trade-offs of {obj} in {domain} are latency, memory footprint, and maintenance."),
    ("List three common mistakes people make with {obj}.",
     "Three common mistakes with {obj}: skipping validation, ignoring edge cases, "
     "and trusting defaults."),
    ("Compare {obj} against the usual alternative in {domain}.",
     "Against the usual {domain} alternative, {obj} wins on throughput and loses on simplicity."),
    ("Give a step-by-step procedure for setting up {obj}.",
     "To set up {obj}: define the inputs, configure the parameters, then verify against "
     "a known case."),
    ("Why would a team choose {obj} over a simpler approach?",
     "A team picks {obj} over something simpler when the simpler approach stops scaling."),
    ("Describe a failure mode of {obj} and how to detect it.",
     "A classic {obj} failure is silent degradation; detect it by monitoring the tail, "
     "not the mean."),
    ("Draft a checklist for reviewing {obj} in a {domain} project.",
     "Reviewing {obj}: check the inputs, check the boundaries, check the rollback path."),
    ("Summarise the history of {obj} in two sentences.",
     "{obj} began as a workaround and became standard practice. Its rough edges are historical."),
)

DOMAINS: tuple[str, ...] = (
    "distributed systems", "data engineering", "numerical computing",
    "compiler design", "network operations", "database internals",
    "embedded firmware", "geospatial analysis", "information retrieval",
    "operations research",
)

OBJECTS: tuple[str, ...] = (
    "a write-ahead log", "a bloom filter", "a consistent hash ring",
    "a columnar store", "a lock-free queue", "an LRU cache",
    "a circuit breaker", "a merkle tree", "a priority queue",
    "an inverted index", "a rate limiter", "a connection pool",
    "a b-tree index", "a vector clock", "a leader election protocol",
    "a copy-on-write buffer", "a token bucket", "a bounded channel",
)

# --- the multilingual slice -------------------------------------------
#
# Small and deliberately hand-written rather than scraped: PUBLIC-only is a
# hard constraint, and short generic sentences authored here carry no
# provenance risk at all. The slice exists to exercise script coverage and
# code-switching in the language-ID filter, not to claim multilingual quality.
#
# Scripts covered: Latin, Cyrillic, Greek, Devanagari, Arabic, Han, Hangul.
#
# Each seed below is a *template* carrying a `{obj}` slot, crossed with the
# per-language term list in MULTILINGUAL_TERMS. Twelve fixed rows would cap the
# slice at twelve records however large the corpus, which made script coverage
# a rounding error -- 6 non-Latin records out of 960 -- and left exactly one
# code-switched row to test the detector with. Crossing seeds with terms keeps
# every record distinct (so no fake duplicates are planted) while letting the
# slice scale with the corpus.

MULTILINGUAL_TERMS: dict[str, tuple[str, ...]] = {
    "es": ("una cola de prioridad", "un filtro de Bloom", "un indice inverso",
           "un limitador de taxa", "un registro de escritura anticipada"),
    "fr": ("un index inverse", "un filtre de Bloom", "une file de priorite",
           "un limiteur de debit", "un journal d'ecriture anticipee"),
    "de": ("ein Schreibprotokoll", "ein Bloom-Filter", "ein invertierter Index",
           "eine Prioritaetswarteschlange", "ein Ratenbegrenzer"),
    "pt": ("um limitador de taxa", "um filtro de Bloom", "um indice invertido",
           "uma fila de prioridade", "um registo de escrita antecipada"),
    "ru": ("журнал предзаписи", "фильтр Блума", "инвертированный индекс",
           "очередь с приоритетом", "ограничитель скорости"),
    "el": ("μια ουρά προτεραιότητας", "ένα φίλτρο Bloom", "ένα ανεστραμμένο ευρετήριο",
           "ένας περιοριστής ρυθμού", "ένα ημερολόγιο προεγγραφής"),
    "hi": ("व्युत्क्रम सूचकांक", "ब्लूम फ़िल्टर", "प्राथमिकता कतार",
           "दर सीमक", "पूर्वलेखन लॉग"),
    "ar": ("مخزن الأعمدة", "مرشح بلوم", "الفهرس المعكوس",
           "طابور الأولوية", "محدد المعدل"),
    "zh": ("一致性哈希环", "布隆过滤器", "倒排索引", "优先队列", "限流器"),
    "ko": ("우선순위 큐", "블룸 필터", "역색인", "속도 제한기", "선행 기록 로그"),
}

# (language, instruction template, response template). The two trailing rows
# are code-switched: two scripts inside one record. Those are the rows that
# break naive language-ID, which is exactly why they are here.
MULTILINGUAL_SEEDS: tuple[tuple[str, str, str], ...] = (
    ("es", "Explica como funciona {obj}.",
     "{obj} ordena los elementos por su importancia relativa en el sistema."),
    ("fr", "Decris brievement le role de {obj}.",
     "{obj} associe chaque terme a la liste des documents qui le contiennent."),
    ("de", "Erklaere kurz, wie {obj} funktioniert.",
     "{obj} speichert Aenderungen, bevor sie auf die Daten angewendet werden."),
    ("pt", "Descreva o funcionamento de {obj}.",
     "{obj} controla quantos pedidos passam por unidade de tempo no sistema."),
    ("ru", "Объясните, как работает {obj}.",
     "{obj} сохраняет изменения до их применения к основным данным."),
    ("el", "Εξηγήστε πώς λειτουργεί {obj}.",
     "{obj} ταξινομεί τα στοιχεία κατά σημασία μέσα στο σύστημα."),
    ("hi", "बताइए कि {obj} कैसे काम करता है।",
     "{obj} प्रत्येक शब्द को उन दस्तावेज़ों से जोड़ता है जिनमें वह आता है।"),
    ("ar", "اشرح كيف يعمل {obj}.",
     "{obj} يخزن البيانات حسب العمود بدلا من الصف في النظام."),
    ("zh", "请解释{obj}的工作原理。",
     "{obj}把键和节点映射到同一个环上，从而减少重新分配。"),
    ("ko", "{obj}가 어떻게 동작하는지 설명하세요.",
     "{obj}는 항목을 중요도 순서로 정렬하여 처리합니다."),
    # Code-switched rows: a Latin technical term embedded in the local script.
    ("es-en", "Explica el trade-off de usar {obj} en production.",
     "El trade-off es que {obj} tiene false positives pero usa poca memoria."),
    ("hi-en", "बताइए कि {obj} को production में कैसे deploy करते हैं।",
     "{obj} को edge पर deploy करना बेहतर होता है क्योंकि latency कम होती है।"),
)

def multilingual_records() -> list[tuple[str, str, str]]:
    """Render every (seed x term) combination into a distinct record.

    Code-switched seeds draw their ``{obj}`` from the *English* object list
    rather than the local term list -- that is what makes them code-switched,
    and it is why they are declared with a hyphenated tag like ``es-en``.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for lang, instruction, response in MULTILINGUAL_SEEDS:
        if "-" in lang:
            base = lang.split("-", 1)[0]
            terms = tuple(o.removeprefix("a ").removeprefix("an ") for o in OBJECTS)
        else:
            base = lang
            terms = MULTILINGUAL_TERMS[base]
        for term in terms:
            pair = (instruction.format(obj=term), response.format(obj=term))
            if pair in seen:
                continue
            seen.add(pair)
            out.append((pair[0], pair[1], lang))
    return out


# --- controlled paraphrase --------------------------------------------
#
# Applied to *planted* near-duplicates. Each rule is a small surface edit that
# leaves the meaning and most of the shingles intact -- which is precisely the
# regime where exact hashing fails and MinHash should not.

_PARAPHRASE_RULES: tuple[tuple[str, str], ...] = (
    ("Explain how", "Could you explain how"),
    ("Write a short summary of", "Write a brief summary of"),
    ("What are the main", "What are the primary"),
    ("List three common", "Name three common"),
    ("Compare", "Contrast"),
    ("Give a step-by-step procedure for", "Give a step by step procedure for"),
    ("Why would a team choose", "Why might a team choose"),
    ("Describe a failure mode of", "Describe one failure mode of"),
    ("Draft a checklist for", "Draft a short checklist for"),
    ("Summarise", "Summarize"),
)


def paraphrase(text: str, rng: random.Random) -> str:
    """A surface-level rewrite that preserves most shingles.

    Not a model paraphrase and not claimed to be one. The job is to produce
    near-duplicates in the band where the detector's behaviour is actually
    interesting: identical enough that a human calls them duplicates, different
    enough that SHA-256 does not.

    **Exactly one edit is applied.** The first version stacked three -- a lead
    rewrite, a "Briefly:" prefix when no rule matched, and a trailing filler
    clause -- applied to the instruction and the response independently. On
    texts this short that is not a paraphrase, it is a rewrite: measured
    Jaccard on the planted pairs came out at 0.26-0.57, entirely below the 0.7
    threshold, so MinHash recall was 0% against pairs that were not actually
    near-duplicates by the stated definition.

    The temptation at that point is to lower the threshold until the number
    looks good. That would be measuring the fixture, not the detector. The
    fixture was wrong: real near-duplicates in generated corpora differ by a
    reworded stem or an appended clause, not by both at once plus a prefix.
    """
    rules = [(src, dst) for src, dst in _PARAPHRASE_RULES if src in text]
    if rules:
        src, dst = rng.choice(rules)
        return text.replace(src, dst, 1)
    # No stem rule applies: append a single short clause instead. Still one
    # edit, and short relative to the text it is appended to.
    return text + rng.choice(
        (" Keep it concise.", " Be specific.", " Assume a technical reader.")
    )


class Generator(Protocol):
    """The swap seam for a real LLM generator.

    An LLM-backed implementation would call a model in ``generate`` and return
    the same ``Record`` objects. Nothing downstream of this module knows or
    cares which implementation produced the corpus -- the hygiene layer is
    generator-agnostic by construction.
    """

    def generate(self, n: int) -> list[Record]: ...


@dataclass(frozen=True, slots=True)
class TemplateGenerator:
    """Deterministic recombinant generator.

    ``multilingual_fraction`` is the share of records drawn from the
    multilingual slice. It defaults low because the slice is small; drawing
    more would just repeat it and inflate the duplicate rate with an artifact
    of the fixture rather than a property of the pipeline.
    """

    seed: int = 0
    multilingual_fraction: float = 0.1

    @staticmethod
    def _distinct_combinations() -> list[tuple[str, str, str]]:
        """Every combination that renders to a *distinct* text.

        The cross product has 1800 cells but far fewer distinct renderings,
        because most task frames never interpolate ``{domain}``: "List three
        common mistakes people make with {obj}" renders identically for all ten
        domains. Sampling distinct *indices* therefore still produced duplicate
        *texts* -- 900 draws collapsed to 541 unique records.

        That mattered for more than corpus size. Ids are content-addressed, so
        duplicate text means duplicate ids, and the planting helpers key their
        ground truth on the id: splicing contamination into one record marked
        every record sharing that id as planted. Ground truth was corrupted by
        a generator detail, which is exactly the kind of coupling that makes a
        measured number quietly wrong rather than obviously broken.

        Enumerating once and deduplicating on the rendered text fixes it at the
        source. 1800 cells is small enough that materialising them is free.

        (An earlier version drew frame/domain/object independently with
        `rng.choice`, which was worse still: birthday collisions turned 600
        draws into 350 distinct texts, and near-duplicate precision collapsed
        to 0.013 because hundreds of records genuinely differed by one noun and
        were correctly flagged while not being *planted*. Sampling without
        replacement from a deduplicated space removes both artifacts.)
        """
        seen: dict[tuple[str, str], tuple[str, str, str]] = {}
        total = len(TASK_FRAMES) * len(DOMAINS) * len(OBJECTS)
        for index in range(total):
            instruction, response, lang = TemplateGenerator._combination_at(index)
            seen.setdefault((instruction, response), (instruction, response, lang))
        return list(seen.values())

    @staticmethod
    def _combination_at(index: int) -> tuple[str, str, str]:
        """Mixed-radix decode of one cross-product cell into rendered text."""
        n_obj, n_dom = len(OBJECTS), len(DOMAINS)
        obj = OBJECTS[index % n_obj]
        index //= n_obj
        domain = DOMAINS[index % n_dom]
        index //= n_dom
        instruction, response = TASK_FRAMES[index % len(TASK_FRAMES)]
        text_i = instruction.format(domain=domain, obj=obj)
        text_r = response.format(domain=domain, obj=obj)
        return text_i[0].upper() + text_i[1:], text_r[0].upper() + text_r[1:], "en"

    @property
    def capacity(self) -> int:
        """How many distinct records this generator can produce.

        Computed from the deduplicated renderings rather than from the raw
        cross-product size, so it is the number a caller can actually request.
        """
        return len(self._distinct_combinations()) + len(multilingual_records())

    def _iter_pairs(self, n: int, rng: random.Random) -> Iterator[tuple[str, str, str]]:
        multi = multilingual_records()
        n_multi = min(int(n * self.multilingual_fraction), len(multi))
        n_template = n - n_multi
        combos = self._distinct_combinations()
        if n_template > len(combos):
            raise ValueError(
                f"requested {n_template} templated records but only {len(combos)} distinct "
                f"renderings exist; add frames, domains or objects rather than allowing "
                f"the generator to repeat itself"
            )
        # Sampled, not sliced: taking the first k rows would take every term
        # for the first few languages and none for the rest, so the script
        # distribution would depend on declaration order.
        for combo in rng.sample(multi, n_multi):
            yield combo
        for combo in rng.sample(combos, n_template):
            yield combo

    def generate(self, n: int) -> list[Record]:
        """Produce ``n`` records with generation provenance attached.

        Every returned record has a distinct id. Ids are content-addressed, so
        that property comes from `_distinct_combinations` guaranteeing distinct
        text -- and it is a property the planting helpers depend on, since they
        key their ground truth on the id. `generate` asserts it rather than
        trusting it: a silent id collision corrupts every measurement
        downstream, and it is cheap to rule out here.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rng = random.Random(self.seed)
        records: list[Record] = []
        for index, (instruction, response, lang) in enumerate(self._iter_pairs(n, rng)):
            rid = content_id(f"{instruction}\n{response}")
            records.append(
                Record(
                    record_id=rid,
                    instruction=instruction,
                    response=response,
                    language=lang,
                    provenance=(),
                ).with_event(
                    STAGE_GENERATE,
                    "template",
                    source="templated",
                    generator="TemplateGenerator",
                    seed=self.seed,
                    index=index,
                    model=None,
                ),
            )
        ids = {r.record_id for r in records}
        if len(ids) != len(records):
            raise AssertionError(
                f"generator produced {len(records)} records with only {len(ids)} distinct "
                f"ids; ground truth keyed on record_id would be corrupted"
            )
        return records


def plant_near_duplicates(
    records: Sequence[Record],
    *,
    n_pairs: int,
    seed: int = 0,
) -> tuple[list[Record], set[tuple[str, str]]]:
    """Append ``n_pairs`` paraphrased copies of existing records.

    Returns the extended corpus and the ground-truth set of duplicate pairs,
    as ``(original_id, copy_id)``. That second value is the whole point: it is
    the answer key the detector is scored against, and it is produced here
    rather than inferred anywhere else.

    Sources are sampled *without replacement* so no original is duplicated
    twice. Overlapping duplicate groups would make pairwise precision/recall
    ambiguous -- one true pair or three? -- and an ambiguous denominator is a
    worse problem than a smaller sample.
    """
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be non-negative, got {n_pairs}")
    if n_pairs > len(records):
        raise ValueError(f"cannot plant {n_pairs} pairs from {len(records)} records")

    rng = random.Random(seed)
    sources = rng.sample(range(len(records)), n_pairs)
    extended = list(records)
    truth: set[tuple[str, str]] = set()

    for idx in sources:
        original = records[idx]
        # Edit one field, not both. Editing both compounds two paraphrases
        # over a ~30-word record and pushes the pair below any sensible
        # near-duplicate threshold -- see `paraphrase` for what that cost.
        if rng.random() < 0.5:
            instruction = paraphrase(original.instruction, rng)
            response = original.response
        else:
            instruction = original.instruction
            response = paraphrase(original.response, rng)
        rid = content_id(f"{instruction}\n{response}")
        if rid == original.record_id:
            # Paraphrase was a no-op. Skip rather than plant a pair the
            # content-addressed id has already merged -- counting it would
            # overstate the planted total against a record that is not there.
            continue
        copy = Record(
            record_id=rid,
            instruction=instruction,
            response=response,
            language=original.language,
            provenance=original.provenance,
        ).with_event(
            STAGE_GENERATE,
            "paraphrase",
            source="templated",
            derived_from=original.record_id,
            planted="near_duplicate",
        )
        extended.append(copy)
        truth.add((original.record_id, rid))

    return extended, truth
