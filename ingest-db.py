#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", str(ROOT / "ayurveda_vector_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "sparsha_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLEAN_CSV = ROOT / "data" / "bhavaprakasha_clean.csv"
REVIEW_CSV = ROOT / "data" / "bhavaprakasha_review.csv"
RAW_DIR = ROOT / "gretil_pipeline" / "raw_texts"
SCHEMA = [
    "herb_sanskrit", "varga_category", "synonyms", "classical_properties",
    "raw_karma_sanskrit", "english_disease", "english_herb", "botanical_name",
    "english_guess", "source_text", "source_verse_ids",
]
VERSE_PATTERNS = {
    "rajanighantu.txt": (
        "Rajanighantu (GRETIL)",
        re.compile(r"(.+?)//\s*(rajni_[A-Za-z0-9.,_*]+)", re.DOTALL),
    ),
    "ashtanga_nighantu.txt": (
        "Ashtanga Nighantu (GRETIL)",
        re.compile(r"(.+?)//\s*(VAnigh_[A-Za-z0-9*]+)", re.DOTALL),
    ),
}


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def stable_id(prefix: str, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def chunks(items: list[dict], size: int = 128) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def schema_document(row: dict[str, str]) -> str:
    labels = {
        "herb_sanskrit": "Herb Sanskrit",
        "varga_category": "Varga category",
        "synonyms": "Synonyms",
        "classical_properties": "Classical properties",
        "raw_karma_sanskrit": "Raw karma Sanskrit",
        "english_disease": "English disease",
        "english_herb": "English herb",
        "botanical_name": "Botanical name",
        "english_guess": "English guess (not authoritative)",
        "source_text": "Source text",
        "source_verse_ids": "Source verse IDs",
    }
    return "\n".join(f"{labels[key]}: {clean(row.get(key))}" for key in SCHEMA)


def load_clean_rows() -> list[dict]:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Run gretil_pipeline/prepare_bhavaprakasha.py first: {CLEAN_CSV}")
    records: list[dict] = []
    with CLEAN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in SCHEMA if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Clean CSV missing columns: {missing}")
        for row in reader:
            normalized = {key: clean(row.get(key)) for key in SCHEMA}
            verse_id = normalized["source_verse_ids"]
            records.append({
                "id": stable_id("bp_safe", verse_id),
                "document": schema_document(normalized),
                "metadata": {
                    "source_text": normalized["source_text"],
                    "source_verse_ids": verse_id,
                    "herb_sanskrit": normalized["herb_sanskrit"],
                    "varga_category": normalized["varga_category"],
                    "record_type": "herb_entry",
                    "review_status": "structurally_accepted",
                    "eligible_for_recommendation": True,
                },
            })
    return records


def load_review_rows() -> list[dict]:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"Missing review CSV: {REVIEW_CSV}")
    records: list[dict] = []
    with REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = clean(row.get("id"))
            # Preserve all original Sanskrit evidence and anomaly reasons. Legacy
            # English/botanical columns are not promoted as verified facts.
            document = "\n".join([
                f"Headword as supplied: {clean(row.get('herb_hindi'))}",
                f"Varga category: {clean(row.get('varga_category'))}",
                f"Synonyms: {clean(row.get('synonyms'))}",
                f"Classical properties: {clean(row.get('classical_properties'))}",
                f"Raw karma Sanskrit: {clean(row.get('raw_karma_sanskrit'))}",
                f"Source record ID: {source_id}",
                f"Blocking anomalies: {clean(row.get('blocking_issues'))}",
                f"Warnings: {clean(row.get('warnings'))}",
            ])
            records.append({
                "id": stable_id("bp_review", source_id),
                "document": document,
                "metadata": {
                    "source_text": "Bhavaprakasha Nighantu (legacy professor dataset; test use)",
                    "source_verse_ids": source_id,
                    "herb_sanskrit": clean(row.get("herb_hindi")),
                    "varga_category": clean(row.get("varga_category")),
                    "record_type": "quarantined_legacy_row",
                    "review_status": "requires_bams_review",
                    "eligible_for_recommendation": False,
                    "blocking_issues": clean(row.get("blocking_issues")),
                },
            })
    return records


def load_raw_verses() -> list[dict]:
    records: list[dict] = []
    for filename, (source_name, pattern) in VERSE_PATTERNS.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        text = path.read_text(encoding="utf-8")
        text = text.split("# Text", 1)[-1]
        for raw_verse, verse_id in pattern.findall(text):
            # Avoid carrying the previous header across the first match.
            verse = re.sub(r"^\s*#+.*$", "", raw_verse, flags=re.MULTILINE).strip()
            if not verse:
                continue
            records.append({
                "id": stable_id("raw_verse", f"{source_name}|{verse_id}"),
                "document": f"Source text: {source_name}\nSource verse ID: {verse_id}\nRaw Sanskrit verse: {verse}",
                "metadata": {
                    "source_text": source_name,
                    "source_verse_ids": verse_id,
                    "herb_sanskrit": "",
                    "varga_category": "",
                    "record_type": "raw_source_verse",
                    "review_status": "unparsed_source_evidence",
                    "eligible_for_recommendation": False,
                },
            })
    return records


def ingest(reset: bool = False) -> None:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if reset:
        existing = {collection.name for collection in client.list_collections()}
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine", "application": "SPARSHA"},
    )

    clean_records = load_clean_rows()
    review_records = load_review_rows()
    raw_records = load_raw_verses()
    records = clean_records + review_records + raw_records
    for number, batch in enumerate(chunks(records), start=1):
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        print(f"Upserted {min(number * 128, len(records))}/{len(records)}", end="\r")

    print("\nIngestion complete")
    print(f"  Chroma path: {DB_PATH}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Recommendation-eligible Bhavaprakasha entries: {len(clean_records)}")
    print(f"  Quarantined Bhavaprakasha rows retained: {len(review_records)}")
    print(f"  Raw GRETIL source verses retained: {len(raw_records)}")
    print(f"  Total collection records: {collection.count()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Rebuild the target collection from scratch")
    return parser.parse_args()


if __name__ == "__main__":
    ingest(reset=parse_args().reset)