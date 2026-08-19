"""SPARSHA FastAPI backend: normalize -> retrieve -> safety verdict -> stream."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator, Literal

import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./ayurveda_vector_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sparsha_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# Retrieve the complete eligible ranking, then safety-check this many candidates
# individually to keep Gemini free-tier usage bounded.
MAX_CLASSIFY_CANDIDATES = max(1, int(os.getenv("MAX_CLASSIFY_CANDIDATES", "12")))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.0"))
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "12")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
GEMINI_RATE_LIMIT_CALLS = max(1, int(os.getenv("GEMINI_RATE_LIMIT_CALLS", "12")))

# Used in addition to vocabulary discovered from the indexed classical records.
KNOWN_SANSKRIT_TERMS = {
    "ajirna", "amavata", "amlapitta", "arsha", "atisara", "chardi", "daha",
    "grahani", "gulma", "hikka", "jvara", "kamala", "kasa", "krimi",
    "kushtha", "mutrakrichra", "pandu", "prameha", "rajayakshma", "raktapitta",
    "shotha", "shula", "svasa", "trishna", "udara", "unmada", "vatarakta",
    "visarpa", "vrana",
}
# Prevent common one-word English queries present in indexed glosses from being
# mistaken for Roman Sanskrit merely because they occur in the corpus.
KNOWN_ENGLISH_CLINICAL_TERMS = {
    "fever", "cough", "headache", "bloating", "bloat", "pain", "diarrhea",
    "constipation", "vomiting", "nausea", "asthma", "anemia", "jaundice",
    "diabetes", "wound", "ulcer", "itching", "swelling", "inflammation",
    "indigestion", "acidity", "fatigue", "weakness", "bleeding",
}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
IAST_RE = re.compile(r"[āīūṛṝḷṅñṭḍṇśṣṃṁḥ]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

app = FastAPI(title="SPARSHA", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self.lock:
            bucket = self.hits[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window - (now - bucket[0])) + 1)
                return False, retry_after
            bucket.append(now)
            return True, 0


class BlockingProviderLimiter:
    """Global limiter for actual Gemini calls, not merely incoming HTTP requests."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self.hits: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.hits and self.hits[0] <= now - self.window:
                    self.hits.popleft()
                if len(self.hits) < self.limit:
                    self.hits.append(now)
                    return
                wait_for = max(0.05, self.window - (now - self.hits[0]) + 0.01)
            time.sleep(wait_for)


limiter = SlidingWindowLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
gemini_limiter = BlockingProviderLimiter(GEMINI_RATE_LIMIT_CALLS)


@app.middleware("http")
async def rate_limit_queries(request: Request, call_next):
    # Limit only AI query traffic, not the UI and health checks.
    if request.url.path in {"/api/query", "/query"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        client_ip = forwarded or (request.client.host if request.client else "unknown")
        allowed, retry_after = limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry shortly."},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


class QueryBody(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class AskRequest(BaseModel):
    # Kept compatible with the existing frontend contract.
    symptom: str = Field(min_length=1, max_length=500)
    language: str = Field(default="en", max_length=10)


class Candidate(BaseModel):
    doc_id: str
    distance: float
    document: str
    metadata: dict[str, Any]


class Verdict(BaseModel):
    status: Literal["SAFE", "DANGER", "NOT_FOUND"]
    reasoning: str
    candidate: Candidate | None = None


_resource_lock = threading.Lock()
_collection = None
_model = None
_index_vocabulary: set[str] | None = None


def get_collection():
    global _collection
    if _collection is None:
        with _resource_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
                embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
                _collection = client.get_collection(
                    name=CHROMA_COLLECTION,
                    embedding_function=embedding_function,
                )
    return _collection


def get_model():
    global _model
    if _model is None:
        with _resource_lock:
            if _model is None:
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY is not configured")
                genai.configure(api_key=api_key)
                _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model


def canonical_token(text: str) -> str:
    return re.sub(r"[^a-zāīūṛṝḷṅñṭḍṇśṣṃṁḥ]", "", text.casefold())


def get_index_vocabulary() -> set[str]:
    """Build a local vocabulary once so Roman Sanskrit already in the DB is not expanded."""
    global _index_vocabulary
    if _index_vocabulary is None:
        with _resource_lock:
            if _index_vocabulary is None:
                result = get_collection().get(include=["documents"])
                vocabulary: set[str] = set(KNOWN_SANSKRIT_TERMS)
                for document in result.get("documents") or []:
                    for token in TOKEN_RE.findall(document or ""):
                        normalized = canonical_token(token)
                        if normalized:
                            vocabulary.add(normalized)
                _index_vocabulary = vocabulary
    return _index_vocabulary


def is_sanskrit_query(query: str) -> bool:
    value = query.strip()
    if DEVANAGARI_RE.search(value) or IAST_RE.search(value):
        return True
    tokens = [canonical_token(token) for token in TOKEN_RE.findall(value)]
    tokens = [token for token in tokens if token]
    if not tokens or len(tokens) > 3:
        return False
    if any(token in KNOWN_ENGLISH_CLINICAL_TERMS for token in tokens):
        return False
    vocabulary = get_index_vocabulary()
    return all(token in vocabulary for token in tokens)


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Gemini did not return valid JSON")
        return json.loads(match.group(1))


def normalize_query(query: str) -> list[str]:
    """Return the raw classical term unchanged; expand likely English only."""
    raw = query.strip()
    if is_sanskrit_query(raw):
        return [raw]

    prompt = f"""You normalize an English clinical query for retrieval from an Ayurvedic
Nighantu corpus. Return ONLY a JSON array of 3 to 8 short search terms. Include the
original key English symptom/disease and likely classical Sanskrit equivalents in
Devanagari and/or standard Roman transliteration. Do not diagnose, prescribe, or
add herbs. Keep terms precise rather than broad.

Query: {raw}"""
    gemini_limiter.acquire()
    response = get_model().generate_content(
        prompt,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    parsed = extract_json(response.text)
    if isinstance(parsed, dict):
        parsed = parsed.get("terms", [])
    terms = [str(term).strip() for term in parsed if str(term).strip()] if isinstance(parsed, list) else []
    # Keep order, include raw query, and avoid an uncontrolled term explosion.
    return list(dict.fromkeys([raw, *terms]))[:8]


def retrieve_candidates(terms: list[str]) -> list[Candidate]:
    """Query every term separately, merge by document ID, and retain min distance."""
    collection = get_collection()
    eligible = collection.get(
        where={"eligible_for_recommendation": True}, include=[]
    ).get("ids", [])
    count = len(eligible)
    if count == 0:
        return []

    merged: dict[str, Candidate] = {}
    for term in terms:
        result = collection.query(
            query_texts=[term],
            n_results=count,  # Full eligible ranked list; safety gate decides where to stop.
            where={"eligible_for_recommendation": True},
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            distance_value = float(distance)
            if distance_value > RETRIEVAL_MAX_DISTANCE:
                continue
            current = merged.get(doc_id)
            if current is None or distance_value < current.distance:
                merged[doc_id] = Candidate(
                    doc_id=doc_id,
                    distance=distance_value,
                    document=document or "",
                    metadata=metadata or {},
                )
    return sorted(merged.values(), key=lambda item: item.distance)


def classify(query: str, candidate: Candidate) -> Verdict:
    prompt = f"""You are a conservative evidence gate for BAMS-qualified Ayurvedic
consultants. Assess ONE retrieved herb record against the consultant's query.
Use only the supplied record, especially raw_karma_sanskrit and classical_properties.

Return ONLY JSON with keys status and reasoning.
- SAFE: the record explicitly supports use for the queried disease/condition.
- DANGER: the record explicitly states harm, aggravation, contraindication, or a
  clearly conflicting action for that condition.
- NOT_FOUND: evidence is absent, ambiguous, merely a synonym match, or requires
  unsupported inference.
Never infer missing facts. Do not turn an uncertain match into SAFE.

Consultant query: {query}
Retrieved record:
{candidate.document}"""
    gemini_limiter.acquire()
    response = get_model().generate_content(
        prompt,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    parsed = extract_json(response.text)
    status = str(parsed.get("status", "NOT_FOUND")).upper()
    if status not in {"SAFE", "DANGER", "NOT_FOUND"}:
        status = "NOT_FOUND"
    reasoning = str(parsed.get("reasoning", "No explicit supporting evidence found.")).strip()
    return Verdict(status=status, reasoning=reasoning, candidate=candidate)


def find_verdict(query: str, candidates: list[Candidate]) -> Verdict:
    """Classify one record at a time; first SAFE wins, otherwise surface DANGER."""
    first_danger: Verdict | None = None
    candidates_to_check = (
        candidates[:MAX_CLASSIFY_CANDIDATES] if MAX_CLASSIFY_CANDIDATES else candidates
    )
    for candidate in candidates_to_check:
        verdict = classify(query, candidate)
        if verdict.status == "SAFE":
            return verdict
        if verdict.status == "DANGER" and first_danger is None:
            first_danger = verdict
    if first_danger is not None:
        return first_danger
    return Verdict(
        status="NOT_FOUND",
        reasoning=(
            "No retrieved record contained explicit support for this query within "
            f"the {len(candidates_to_check)} candidate(s) safety-checked."
        ),
    )


def sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def stream_safe_record(query: str, verdict: Verdict) -> Iterator[str]:
    yield sse({"status": verdict.status, "reasoning": verdict.reasoning})
    if verdict.status != "SAFE" or verdict.candidate is None:
        yield sse_done()
        return

    candidate = verdict.candidate
    prompt = f"""For a BAMS-qualified consultant, present the selected herb's classical
properties from the supplied record. Stay strictly grounded in the record. Preserve
Sanskrit exactly; do not invent translations, dosage, formulations, indications,
contraindications, botanical identity, or English names. Clearly identify the herb,
source text, verse IDs, classical properties, and raw karma when available. Mention
that english_guess is non-authoritative if you include it.

Consultant query: {query}
Verified record:
{candidate.document}"""
    gemini_limiter.acquire()
    response = get_model().generate_content(
        prompt,
        generation_config={"temperature": 0.1},
        stream=True,
    )
    for chunk in response:
        text = getattr(chunk, "text", "")
        if text:
            yield sse({"text": text})
    yield sse_done()


def query_events(query: str) -> Iterator[str]:
    try:
        terms = normalize_query(query)
        candidates = retrieve_candidates(terms)
        verdict = find_verdict(query, candidates) if candidates else Verdict(
            status="NOT_FOUND", reasoning="The vector database returned no candidate records."
        )
        yield from stream_safe_record(query, verdict)
    except Exception as exc:
        yield sse({"status": "NOT_FOUND", "reasoning": f"Backend error: {exc}"})
        yield sse_done()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        count = get_collection().count()
        return {"status": "ok", "collection": CHROMA_COLLECTION, "documents": count}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.post("/api/query")
def query_post(body: QueryBody) -> StreamingResponse:
    return StreamingResponse(
        query_events(body.query.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/query")
def query_get(q: str = Query(min_length=1, max_length=500)) -> StreamingResponse:
    # GET support keeps native EventSource frontends compatible.
    return StreamingResponse(
        query_events(q.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
def ask(body: AskRequest) -> StreamingResponse:
    """Compatibility endpoint used by the supplied frontend."""
    return StreamingResponse(
        query_events(body.symptom.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
