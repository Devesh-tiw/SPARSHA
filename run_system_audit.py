#!/usr/bin/env python3
"""Offline, reproducible SPARSHA corpus/mapping/safety audit.

This does not claim clinical validation. It measures structural integrity,
terminology coverage, explicit Sanskrit evidence coverage, and safety quarantine.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLEAN = DATA / "bhavaprakasha_clean.csv"
REVIEW = DATA / "bhavaprakasha_review.csv"
TERM_MAP = DATA / "clinical_term_map.json"
MATRIX = DATA / "automated_condition_test_matrix.csv"
REPORT = ROOT / "SPARSHA_TEST_REPORT.md"

POSITIVE = re.compile(r"(?:हर|हरी|हृत्|घ्न|नाश|जित्|प्रणुत्|अपह|शमन)")
DANGER = re.compile(r"(?:कर|कृत्|प्रद|वर्धन|कोपन|विदाह|मृत्यु|गर्भपात)")
HIGH_RISK_NAMES = (
    "अहिफेन", "धत्तूर", "जयपाल", "कुपीलु", "वत्सनाभ", "करवीर",
    "गुञ्जा", "भल्लातक", "विषमुष्टि",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_quarantined(row: dict[str, str]) -> bool:
    match = re.fullmatch(r"BPN_(\d+)", row["source_verse_ids"])
    number = int(match.group(1)) if match else -1
    return 589 <= number <= 695 or any(name in row["herb_sanskrit"] for name in HIGH_RISK_NAMES)


def action_evidence(term: str, karma: str) -> tuple[bool, bool]:
    positive = danger = False
    for match in re.finditer(re.escape(term), karma):
        window = karma[match.start() : match.end() + 28]
        positive |= bool(POSITIVE.search(window))
        danger |= bool(DANGER.search(window))
    return positive, danger


def main() -> None:
    clean = read_csv(CLEAN)
    review = read_csv(REVIEW)
    terms = json.loads(TERM_MAP.read_text(encoding="utf-8"))["terms"]
    quarantined_ids = {row["source_verse_ids"] for row in clean if is_quarantined(row)}
    eligible = [row for row in clean if row["source_verse_ids"] not in quarantined_ids]

    matrix_rows = []
    for entry in terms:
        term = entry["sanskrit_devanagari"]
        mentions, positives, dangers = [], [], []
        for row in eligible:
            karma = row["raw_karma_sanskrit"]
            if term not in karma:
                continue
            mentions.append(row["source_verse_ids"])
            positive, danger = action_evidence(term, karma)
            if positive:
                positives.append(row["source_verse_ids"])
            if danger:
                dangers.append(row["source_verse_ids"])
        if positives:
            result = "EXPLICIT_ACTION_EVIDENCE"
        elif mentions:
            result = "MENTION_REQUIRES_LLM_OR_BAMS_REVIEW"
        else:
            result = "CORPUS_DATA_GAP"
        matrix_rows.append({
            "english_name": entry["english_name"],
            "sanskrit_roman": entry["sanskrit_roman"],
            "sanskrit_devanagari": term,
            "verification_status": entry["verification_status"],
            "test_result": result,
            "eligible_mentions": len(set(mentions)),
            "explicit_positive_records": len(set(positives)),
            "nearby_danger_records": len(set(dangers)),
            "example_citations": "|".join(sorted(set(positives or mentions))[:5]),
        })

    with MATRIX.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    counts = {key: sum(row["test_result"] == key for row in matrix_rows) for key in (
        "EXPLICIT_ACTION_EVIDENCE", "MENTION_REQUIRES_LLM_OR_BAMS_REVIEW", "CORPUS_DATA_GAP"
    )}
    gaps = [f"{row['english_name']} (`{row['sanskrit_devanagari']}`)" for row in matrix_rows if row["test_result"] == "CORPUS_DATA_GAP"]
    review_only = [f"{row['english_name']} (`{row['sanskrit_devanagari']}`)" for row in matrix_rows if row["test_result"] == "MENTION_REQUIRES_LLM_OR_BAMS_REVIEW"]

    report = f"""# SPARSHA Automated Test Report

## Scope

This is an automated structural and evidence-coverage audit. It is **not clinical validation** and does not establish 95% medical accuracy. Final mappings and verdicts require BAMS review and an independently labelled gold test set.

## Dataset integrity

- Original Bhavaprakasha rows retained: **{len(clean) + len(review)}**
- Structurally accepted rows before toxicity policy: **{len(clean)}**
- Structurally quarantined rows: **{len(review)}**
- High-risk, mineral, poison, or purification-dependent accepted rows newly quarantined: **{len(quarantined_ids)}**
- Recommendation-eligible rows after safety policy: **{len(eligible)}**
- Terminology entries tested: **{len(matrix_rows)}**

## Terminology/evidence test results

- Explicit nearby Sanskrit action evidence: **{counts['EXPLICIT_ACTION_EVIDENCE']}**
- Corpus mention, but needs Gemini/BAMS interpretation: **{counts['MENTION_REQUIRES_LLM_OR_BAMS_REVIEW']}**
- No eligible corpus evidence: **{counts['CORPUS_DATA_GAP']}**

### Corpus data gaps

{chr(10).join('- ' + item for item in gaps) or '- None'}

### Mention-only terms requiring review

{chr(10).join('- ' + item for item in review_only) or '- None'}

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
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {MATRIX}")
    print(f"Wrote {REPORT}")
    print(json.dumps({"eligible": len(eligible), "quarantined_high_risk": len(quarantined_ids), **counts}, indent=2))


if __name__ == "__main__":
    main()
