# Bhāvaprakāśa dataset audit

## Result

- Input rows: **1110**
- Accepted into high-confidence test CSV: **726**
- Quarantined for manual BAMS review: **384**
- Structural acceptance rate: **65.41%**
- Duplicate source IDs: **0**
- Exact duplicate-content groups: **0**

## Blocking anomalies

- `MULTI_ENTITY_OR_GROUP_HEADWORD`: 234
- `COMMENTARY_AS_HEADWORD`: 142
- `HEADWORD_TOO_LONG`: 101
- `NO_KARMA_FOR_RECOMMENDATION`: 73
- `NO_PROPERTIES_OR_KARMA`: 56
- `CONTEXT_DEPENDENT_HEADWORD`: 42
- `MISSING_HEADWORD`: 21
- `VERSE_OR_MULTILINE_HEADWORD`: 15

## Non-blocking warnings

- `MISSING_CLASSICAL_PROPERTIES`: 197
- `LEGACY_HERB_ENGLISH_CONTAINS_SANSKRIT_TEXT`: 100
- `UNVERIFIED_LEGACY_BOTANICAL_NAME`: 25
- `UNVERIFIED_LEGACY_ENGLISH_HERB`: 1

## Safety decision

English disease, English herb, botanical name, and English guess were left blank. Existing legacy values were not treated as verified because several `herb_english` cells contain Sanskrit verses rather than English names. No quarantined row is eligible for recommendation retrieval until reviewed by a BAMS expert.

This audit measures structural consistency only. It does **not** establish 95% clinical accuracy. That requires an expert-labelled gold test set with disease-to-record relevance and contraindication verdicts.
