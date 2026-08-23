# SPARSHA Automated Test Report

## Scope

This is an automated structural and evidence-coverage audit. It is **not clinical validation** and does not establish 95% medical accuracy. Final mappings and verdicts require BAMS review and an independently labelled gold test set.

## Dataset integrity

- Original Bhavaprakasha rows retained: **1110**
- Structurally accepted rows before toxicity policy: **726**
- Structurally quarantined rows: **384**
- High-risk, mineral, poison, or purification-dependent accepted rows newly quarantined: **42**
- Recommendation-eligible rows after safety policy: **684**
- Terminology entries tested: **56**

## Terminology/evidence test results

- Explicit nearby Sanskrit action evidence: **42**
- Corpus mention, but needs Gemini/BAMS interpretation: **5**
- No eligible corpus evidence: **9**

### Corpus data gaps

- inflammatory arthritis-like disorder (`आमवात`)
- goiter / neck swelling (`गलघण्ड`)
- fistula-in-ano (`भगन्दर`)
- diabetes-like Prameha subtype (`मधुमेह`)
- urinary disorder (`मेहरोग`)
- consumptive syndrome (`राजयक्ष्मा`)
- gout-like Vata-Rakta disorder (`वातरक्त`)
- headache / head pain (`शिरःशूल`)
- degenerative joint disorder (`सन्धिवात`)

### Mention-only terms requiring review

- abdominal bloating / distension (`आध्मान`)
- severe mental disturbance (`उन्माद`)
- eye-disease group (`नेत्ररोग`)
- rhinitis / nasal catarrh (`प्रतिश्याय`)
- dysentery-like disorder (`प्रवाहिका`)

## Runtime observations already demonstrated

- `Jvara/jwara/fever` reached an explicit cited record in approximately **0.28–0.30 seconds** after warm-up.
- Misspelled `bloting` was mapped to bloating/`आध्मान` and returned cited `लवङ्गम्`, `BPN_0164`.
- `headache` mapped to `शिरःशूल` and correctly returned `NOT_FOUND` because it is a corpus data gap.
- Friendly `hello` conversation and browser TTS controls functioned in the supplied UI screenshots.

## Important anomalies found

1. The legacy Varga assignment incorrectly places a metals/minerals block under `आम्रादिफलवर्गः`.
2. Toxic and purification-dependent entries were structurally accepted by the first cleaner; the updated ingestion policy now marks them ineligible.
3. Blank English and botanical fields are expected and must remain hidden rather than guessed.
4. Classical terms are not always one-to-one biomedical diagnoses; all current English mappings remain `PROVISIONAL_TEST_ONLY`.
5. One herb may explicitly match several conditions; this is evidence retrieval, not individualized prescribing.

## Required next validation

- Rebuild ChromaDB with the updated high-risk quarantine policy.
- Run end-to-end API tests after rebuilding.
- Have a BAMS reviewer approve the terminology map and a gold test set.
- Measure SAFE precision, DANGER recall, citation correctness, false-SAFE rate, and p50/p95 latency.

Detailed per-term results: `data/automated_condition_test_matrix.csv`.
