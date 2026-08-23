#!/usr/bin/env python3
"""Conservatively migrate and audit the 1,110-row Bhāvaprakāśa test dataset.

No translation or medical fact is generated. Ambiguous rows are quarantined for
BAMS review rather than entering the recommendation collection.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "Master_Ayurveda_Database_Fixed.csv"
CLEAN = ROOT / "data" / "bhavaprakasha_clean.csv"
REVIEW = ROOT / "data" / "bhavaprakasha_review.csv"
REPORT_JSON = ROOT / "data" / "bhavaprakasha_audit.json"
REPORT_MD = ROOT / "data" / "bhavaprakasha_audit.md"

INPUT_COLUMNS = [
    "id", "herb_hindi", "synonyms", "herb_english", "botanical_name",
    "varga_category", "classical_properties", "raw_karma_sanskrit",
    "english_symptoms", "hindi_symptoms",
]
OUTPUT_COLUMNS = [
    "herb_sanskrit", "varga_category", "synonyms", "classical_properties",
    "raw_karma_sanskrit", "english_disease", "english_herb", "botanical_name",
    "english_guess", "source_text", "source_verse_ids",
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
ID_RE = re.compile(r"^BPN_\d{4}$")
COMMENTARY_RE = re.compile(
    r"(?:^अथ|^इति|^उक्त|^तत्र|^तस्य|^तेषां|^एतद|^एवं|^प्रथम|^द्वितीय|"
    r"गुणानाह|नामान्याह|नामानि|लक्षणमाह|दोषानाह|उत्पत्तिमाह|इत्याह|कथं)"
)
VERSE_MARK_RE = re.compile(r"[|।॥]|\|\||\n|\r|\t|[०-९]{1,3}")
GROUP_RE = re.compile(
    r"(?:द्वयम्|द्वयं|त्रयम्|त्रयं|चतुष्टय|पञ्चक|षट्क|सप्तक|अष्टक|नवक|दशक|"
    r"वर्गः|वर्गम्|गणः|गणम्|समूह|आदीनां|शाकानि)"
)
GENERIC_CONTEXT_HEADWORDS = {
    "तत्फलम्", "फलम्", "पत्रम्", "बीजम्", "मूलम्", "पुष्पम्", "त्वक्",
    "मज्जा", "निर्यासः", "पक्वम्", "पक्वा", "अपक्वम्", "बालम्", "वृद्ध",
    "स्वादु", "वर्षोषितं", "आर्द्रम्", "शुष्कम्", "नवीनम्", "पुराणम्",
}


def value(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def contains_sanskrit(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def inspect_row(row: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (blocking issues, warning-only issues)."""
    blocking: list[str] = []
    warnings: list[str] = []
    row_id = value(row, "id")
    herb = value(row, "herb_hindi")
    props = value(row, "classical_properties")
    karma = value(row, "raw_karma_sanskrit")
    synonyms = value(row, "synonyms")

    if not ID_RE.fullmatch(row_id):
        blocking.append("INVALID_SOURCE_ID")
    if not herb:
        blocking.append("MISSING_HEADWORD")
    else:
        if not contains_sanskrit(herb):
            blocking.append("HEADWORD_NOT_DEVANAGARI")
        if COMMENTARY_RE.search(herb):
            blocking.append("COMMENTARY_AS_HEADWORD")
        if VERSE_MARK_RE.search(herb):
            blocking.append("VERSE_OR_MULTILINE_HEADWORD")
        # Recommendation records need an independently identifiable dravya.
        # Multiword, parenthesized, hyphenated, or very long labels are retained
        # for review because many are variants, groups, or contextual phrases.
        if len(herb) > 24:
            blocking.append("HEADWORD_TOO_LONG")
        if re.search(r"[\s,;()\[\]{}\-–—/:]", herb) or GROUP_RE.search(herb):
            blocking.append("MULTI_ENTITY_OR_GROUP_HEADWORD")
        if herb in GENERIC_CONTEXT_HEADWORDS:
            blocking.append("CONTEXT_DEPENDENT_HEADWORD")

    critical_text = " ".join((herb, synonyms, props, karma))
    if "�" in critical_text:
        blocking.append("CORRUPT_UNICODE_REPLACEMENT_CHARACTER")
    if not props and not karma:
        blocking.append("NO_PROPERTIES_OR_KARMA")
    elif not karma:
        # Properties alone can describe pharmacodynamics but cannot establish an
        # explicit disease indication for the SAFE gate.
        blocking.append("NO_KARMA_FOR_RECOMMENDATION")
    elif not props:
        warnings.append("MISSING_CLASSICAL_PROPERTIES")

    # These columns came from the earlier professor-data workflow. They are not
    # trusted as verified mappings and are deliberately not copied to clean data.
    herb_en = value(row, "herb_english")
    botanical = value(row, "botanical_name")
    if herb_en:
        warnings.append(
            "LEGACY_HERB_ENGLISH_CONTAINS_SANSKRIT_TEXT"
            if contains_sanskrit(herb_en) else "UNVERIFIED_LEGACY_ENGLISH_HERB"
        )
    if botanical:
        warnings.append("UNVERIFIED_LEGACY_BOTANICAL_NAME")
    if value(row, "english_symptoms"):
        warnings.append("UNVERIFIED_LEGACY_ENGLISH_SYMPTOMS")
    if value(row, "hindi_symptoms"):
        warnings.append("UNVERIFIED_LEGACY_HINDI_SYMPTOMS")

    return list(dict.fromkeys(blocking)), list(dict.fromkeys(warnings))


def migrate(row: dict[str, str]) -> dict[str, str]:
    # English and botanical fields intentionally remain blank until independently
    # verified. herb_hindi is actually the Sanskrit headword in Devanagari.
    return {
        "herb_sanskrit": value(row, "herb_hindi"),
        "varga_category": value(row, "varga_category"),
        "synonyms": value(row, "synonyms"),
        "classical_properties": value(row, "classical_properties"),
        "raw_karma_sanskrit": value(row, "raw_karma_sanskrit"),
        "english_disease": "",
        "english_herb": "",
        "botanical_name": "",
        "english_guess": "",
        "source_text": "Bhavaprakasha Nighantu (legacy professor dataset; test use)",
        "source_verse_ids": value(row, "id"),
    }


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in INPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing input columns: {', '.join(missing)}")
        rows = list(reader)

    clean_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    blocking_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    id_counts: Counter[str] = Counter(value(row, "id") for row in rows)
    content_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for row in rows:
        blocking, warnings = inspect_row(row)
        blocking_counts.update(blocking)
        warning_counts.update(warnings)
        signature = tuple(value(row, column) for column in INPUT_COLUMNS if column != "id")
        content_ids[signature].append(value(row, "id"))

        if blocking:
            review_rows.append({
                **{column: value(row, column) for column in INPUT_COLUMNS},
                "blocking_issues": "|".join(blocking),
                "warnings": "|".join(warnings),
            })
        else:
            clean_rows.append(migrate(row))

    exact_duplicate_groups = [ids for ids in content_ids.values() if len(ids) > 1]
    duplicate_ids = sorted(row_id for row_id, count in id_counts.items() if count > 1)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(clean_rows)
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as handle:
        review_columns = INPUT_COLUMNS + ["blocking_issues", "warnings"]
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()
        writer.writerows(review_rows)

    report = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "total_rows": len(rows),
        "accepted_high_confidence_rows": len(clean_rows),
        "quarantined_rows": len(review_rows),
        "acceptance_percent": round(100 * len(clean_rows) / max(1, len(rows)), 2),
        "blocking_issue_counts": dict(blocking_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "duplicate_source_ids": duplicate_ids,
        "exact_duplicate_content_groups": exact_duplicate_groups,
        "medical_accuracy_claim": (
            "No clinical-accuracy percentage is claimed. This is structural data validation; "
            "quarantined rows require BAMS review and clinical validation requires a gold test set."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    issue_lines = "\n".join(f"- `{name}`: {count}" for name, count in blocking_counts.most_common()) or "- None"
    warning_lines = "\n".join(f"- `{name}`: {count}" for name, count in warning_counts.most_common()) or "- None"
    REPORT_MD.write_text(
        f"""# Bhāvaprakāśa dataset audit

## Result

- Input rows: **{len(rows)}**
- Accepted into high-confidence test CSV: **{len(clean_rows)}**
- Quarantined for manual BAMS review: **{len(review_rows)}**
- Structural acceptance rate: **{report['acceptance_percent']}%**
- Duplicate source IDs: **{len(duplicate_ids)}**
- Exact duplicate-content groups: **{len(exact_duplicate_groups)}**

## Blocking anomalies

{issue_lines}

## Non-blocking warnings

{warning_lines}

## Safety decision

English disease, English herb, botanical name, and English guess were left blank. Existing legacy values were not treated as verified because several `herb_english` cells contain Sanskrit verses rather than English names. No quarantined row is eligible for recommendation retrieval until reviewed by a BAMS expert.

This audit measures structural consistency only. It does **not** establish 95% clinical accuracy. That requires an expert-labelled gold test set with disease-to-record relevance and contraindication verdicts.
""",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {CLEAN}\nWrote {REVIEW}\nWrote {REPORT_MD}")


if __name__ == "__main__":
    main()
