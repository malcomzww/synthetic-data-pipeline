# Inventory coverage

Anchored to `Bucket_Concept_Inventory.md`, bucket B1 (LLM & post-training).

- 1C synthetic data generation: self-instruct, evol-instruct, persona, Magpie-style self-prompting
- 1C synthetic risks: model collapse, distribution narrowing, style leakage, benchmark contamination
- 1C deduplication: exact, MinHash/LSH, semantic
- 1C decontamination via n-gram overlap
- 1C filtering: quality classifier, perplexity, toxicity, language ID, PII/PHI scrubbing
- 1C multilingual and low-resource: script coverage, transliteration, code-switching
- 1C data versioning, lineage, licensing (TRY-M)
