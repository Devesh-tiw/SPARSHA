#!/usr/bin/env python3
"""Build a corpus-derived Sanskrit/English clinical terminology review list.

No term produced here is automatically BAMS-approved. English labels are
provisional test labels; blank labels require expert mapping. Runtime code must
use only statuses explicitly allowed for the current environment.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "bhavaprakasha_clean.csv"
RAJ = ROOT / "gretil_pipeline" / "raw_texts" / "rajanighantu.txt"
OUT = ROOT / "data" / "disease_lexicon_all.csv"
REVIEW = ROOT / "data" / "disease_lexicon_review.csv"
RUNTIME_MAP = ROOT / "data" / "clinical_term_map.json"

# Conservative labels: where biomedical equivalence is imperfect, the label
# explicitly says "group", "syndrome", or "-like" instead of claiming identity.
SEEDS: dict[str, tuple[str, str, str, str]] = {
    "ज्वर": ("Jvara", "fever", "pyrexia|febrile illness", "Broad classical fever category"),
    "कास": ("Kasa", "cough", "coughing disorder", "Not every cough has the same doshic diagnosis"),
    "श्वास": ("Shvasa", "dyspnea / breathing disorder", "breathlessness|asthma-like disorder", "Not identical to asthma in every context"),
    "प्रमेह": ("Prameha", "urinary-metabolic disorder group", "polyuria|metabolic disorder", "Not automatically equivalent to diabetes"),
    "मधुमेह": ("Madhumeha", "diabetes-like Prameha subtype", "diabetes mellitus", "Biomedical equivalence requires clinical review"),
    "पाण्डु": ("Pandu", "anemia-like disorder", "pallor disorder", "Not automatically equivalent to laboratory-confirmed anemia"),
    "कामला": ("Kamala", "jaundice-like disorder", "jaundice", "Classical category; etiology may differ"),
    "अतिसार": ("Atisara", "diarrhea", "loose stools", "Broad classical category"),
    "ग्रहणी": ("Grahani", "Grahani disorder / malabsorption syndrome", "chronic bowel dysfunction", "Not identical to IBS"),
    "अर्श": ("Arsha", "hemorrhoids", "piles", "Classical Arsha category"),
    "कुष्ठ": ("Kushtha", "classical skin-disease group", "skin disorder|dermatosis", "Must never be translated automatically as leprosy"),
    "श्वित्र": ("Shvitra", "depigmentation disorder", "vitiligo-like disorder", "Clinical confirmation required"),
    "शोथ": ("Shotha", "swelling / edema", "inflammation|oedema", "Context determines edema versus inflammation"),
    "शूल": ("Shula", "pain / colic", "ache|colicky pain", "Site must be specified"),
    "अम्लपित्त": ("Amlapitta", "acid-peptic / hyperacidity-like disorder", "acidity|acid dyspepsia", "Not a single biomedical diagnosis"),
    "अजीर्ण": ("Ajirna", "indigestion", "dyspepsia", "Classical digestive category"),
    "छर्दि": ("Chardi", "vomiting", "emesis", "Symptom/category"),
    "हिक्का": ("Hikka", "hiccup", "hiccups", "Classical category"),
    "कृमि": ("Krimi", "worm / parasitic disorder", "intestinal worms|parasitosis", "May include broader classical concepts"),
    "व्रण": ("Vrana", "wound / ulcer", "wound|ulcer", "Context must distinguish wound and ulcer"),
    "विसर्प": ("Visarpa", "spreading inflammatory skin disorder", "erysipelas-like disorder", "Not automatically equivalent to erysipelas"),
    "रक्तपित्त": ("Raktapitta", "classical bleeding disorder", "bleeding tendency|hemorrhagic disorder", "Broad classical category"),
    "वातरक्त": ("Vatarakta", "gout-like Vata-Rakta disorder", "gout", "Not automatically laboratory-confirmed gout"),
    "आमवात": ("Amavata", "inflammatory arthritis-like disorder", "rheumatoid arthritis-like disorder", "Not automatically equivalent to rheumatoid arthritis"),
    "सन्धिवात": ("Sandhivata", "degenerative joint disorder", "osteoarthritis-like disorder|joint pain", "Requires site and clinical assessment"),
    "राजयक्ष्मा": ("Rajayakshma", "consumptive syndrome", "tuberculosis-like wasting disorder", "Not automatically microbiologically confirmed tuberculosis"),
    "क्षय": ("Kshaya", "wasting / depletion disorder", "consumption|tissue depletion", "Context dependent"),
    "मूत्रकृच्छ्र": ("Mutrakrichhra", "dysuria", "painful urination", "Symptom/category"),
    "अश्मरी": ("Ashmari", "urinary calculus", "urinary stone|kidney-stone-like disorder", "Anatomical site requires confirmation"),
    "गुल्म": ("Gulma", "abdominal mass/distension syndrome", "abdominal lump|abdominal distension", "Not automatically a tumor"),
    "उदर": ("Udara", "abdominal disorder group", "abdominal disease", "Too broad without subtype"),
    "आनाह": ("Anaha", "obstruction / abdominal distension", "constipation|abdominal obstruction", "Context dependent"),
    "आध्मान": ("Adhmana", "abdominal bloating / distension", "bloating|flatulence", "Symptom/category"),
    "तृष्णा": ("Trishna", "excessive thirst", "thirst|polydipsia", "Symptom/category"),
    "दाह": ("Daha", "burning sensation", "burning", "Site must be specified"),
    "उन्माद": ("Unmada", "severe mental disturbance", "psychosis-like disorder|mental disturbance", "Not a single psychiatric diagnosis"),
    "अपस्मार": ("Apasmara", "seizure / loss-of-consciousness disorder", "epilepsy-like disorder|seizure", "Not automatically equivalent to epilepsy"),
    "हृद्रोग": ("Hridroga", "heart-disease group", "cardiac disorder", "Broad classical category"),
    "भगन्दर": ("Bhagandara", "fistula-in-ano", "anal fistula", "Classical category"),
    "श्लीपद": ("Shlipada", "elephantiasis-like limb swelling", "elephantiasis|lymphedema-like disorder", "Etiology requires confirmation"),
    "कण्डू": ("Kandu", "itching", "pruritus", "Symptom/category"),
    "प्रतिश्याय": ("Pratishyaya", "rhinitis / nasal catarrh", "runny nose|rhinitis", "Classical category"),
    "पीनस": ("Pinasa", "chronic nasal/sinus disorder", "chronic rhinitis|sinusitis-like disorder", "Not automatically bacterial sinusitis"),
    "मन्दाग्नि": ("Mandagni", "low digestive capacity", "poor digestion|low appetite", "Ayurvedic functional concept"),
    "अरुचि": ("Aruchi", "loss of appetite / taste", "anorexia|poor appetite", "Not psychiatric anorexia nervosa"),
    "भ्रम": ("Bhrama", "vertigo / dizziness", "dizziness|vertigo", "Symptom/category"),
    "मूर्च्छा": ("Murchha", "fainting / syncope-like state", "syncope|fainting", "Urgent biomedical causes must be excluded"),
    "स्थौल्य": ("Sthaulya", "obesity / excessive adiposity", "obesity", "Clinical assessment required"),
    "मेहरोग": ("Meharoga", "urinary disorder", "urinary disease", "Broad category"),
    "नेत्ररोग": ("Netraroga", "eye-disease group", "eye disorder", "Broad category"),
    "मुखरोग": ("Mukharoga", "oral-disease group", "mouth disorder|oral disease", "Broad category"),
    "गलRequest": ("", "", "", ""),  # removed below; guards accidental editing
    "गलघण्ड": ("Galaganda", "goiter / neck swelling", "goitre|neck swelling", "Not every neck swelling is thyroid disease"),
    "विद्रधि": ("Vidradhi", "abscess-like inflammatory swelling", "abscess", "Clinical confirmation required"),
    "प्रवाहिका": ("Pravahika", "dysentery-like disorder", "dysentery|tenesmus", "Not automatically infectious dysentery"),
    "विबन्ध": ("Vibandha", "constipation / obstruction", "constipation", "Context dependent"),
}
SEEDS.pop("गलRequest", None)

ACTION_SUFFIXES = (
    "विनाशिनी", "विनाशन", "निवारिणी", "निवारण", "प्रणाशन", "नाशिनी", "नाशन",
    "हन्त्री", "हन्ति", "हरिणी", "हारिणी", "हरी", "हर", "हृत्", "घ्नी", "घ्न",
    "जित्", "प्रणुत्", "अपह", "शमन", "करिणी", "कारिणी", "कर", "कृत्", "प्रद",
    "वर्धन", "कोपन",
)
STOP_BASES = {
    "दीप", "पाचन", "बल", "वर्ण", "शुक्र", "वृष्य", "रुचि", "ग्राहि", "लेखन",
    "रोचन", "सार", "लघु", "गुरु", "स्निग्ध", "रूक्ष", "मेध्य", "चक्षुष्य",
}
DEV_TOKEN = re.compile(r"[\u0900-\u097f]+")


def norm(text: str) -> str:
    return re.sub(r"[\s\-–—,;।|()'\"]+", "", text.strip())


def derive_candidate(token: str) -> str | None:
    token = norm(token)
    if token in SEEDS:
        return token
    for suffix in ACTION_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            base = token[: -len(suffix)]
            if base not in STOP_BASES and len(base) >= 3:
                return base
    return None


def main() -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing {CLEAN}")

    citations: dict[str, set[str]] = defaultdict(set)
    source_examples: dict[str, list[str]] = defaultdict(list)
    candidates: set[str] = set(SEEDS)

    with CLEAN.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            karma = (row.get("raw_karma_sanskrit") or "").strip()
            citation = (row.get("source_verse_ids") or "").strip()
            for token in DEV_TOKEN.findall(karma):
                candidate = derive_candidate(token)
                if not candidate:
                    continue
                candidates.add(candidate)
                if citation:
                    citations[candidate].add(citation)
                if karma and len(source_examples[candidate]) < 2:
                    source_examples[candidate].append(karma[:240])

    # Add exact seed occurrences and citations even when written without an action suffix.
    with CLEAN.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = " ".join((row.get("raw_karma_sanskrit") or "", row.get("classical_properties") or ""))
            citation = (row.get("source_verse_ids") or "").strip()
            for term in SEEDS:
                if term in text and citation:
                    citations[term].add(citation)

    fields = [
        "sanskrit_devanagari", "sanskrit_roman", "english_name", "english_aliases",
        "classical_synonyms", "source_text", "source_verse_ids", "source_example",
        "ambiguity_notes", "verification_status", "verified_by",
    ]
    rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    for term in sorted(candidates):
        seed = SEEDS.get(term)
        if seed:
            roman, english, aliases, ambiguity = seed
            status = "PROVISIONAL_TEST_ONLY"
        else:
            roman = english = aliases = ""
            ambiguity = "Automatically extracted candidate; verify that this is a disease term and not an action/quality."
            status = "BAMS_REVIEW_REQUIRED"
        row = {
            "sanskrit_devanagari": term,
            "sanskrit_roman": roman,
            "english_name": english,
            "english_aliases": aliases,
            "classical_synonyms": "",
            "source_text": "Bhavaprakasha Nighantu (legacy professor dataset; test use)",
            "source_verse_ids": "|".join(sorted(citations.get(term, set()))),
            "source_example": " || ".join(source_examples.get(term, [])),
            "ambiguity_notes": ambiguity,
            "verification_status": status,
            "verified_by": "",
        }
        rows.append(row)
        if status == "BAMS_REVIEW_REQUIRED":
            review_rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, output_rows in ((OUT, rows), (REVIEW, review_rows)):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    runtime_entries = []
    for term, (roman, english, aliases, ambiguity) in sorted(SEEDS.items()):
        runtime_entries.append({
            "sanskrit_devanagari": term,
            "sanskrit_roman": roman,
            "english_name": english,
            "english_aliases": [item.strip() for item in aliases.split("|") if item.strip()],
            "ambiguity_notes": ambiguity,
            "verification_status": "PROVISIONAL_TEST_ONLY",
        })
    RUNTIME_MAP.write_text(
        json.dumps({"terms": runtime_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} total terminology rows: {OUT}")
    print(f"Provisional mapped terms: {len(SEEDS)}")
    print(f"BAMS-review-required extracted candidates: {len(review_rows)}")
    print(f"Wrote provisional runtime map: {RUNTIME_MAP}")
    print("No row is marked BAMS_APPROVED automatically.")


if __name__ == "__main__":
    main()
