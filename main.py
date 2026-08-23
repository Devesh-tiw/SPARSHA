"""SPARSHA FastAPI backend: fast retrieval -> isolated safety gate -> SSE output."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Literal

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

CHROMA_PATH = os.getenv("CHROMA_PERSIST_PATH", str(ROOT / "ayurveda_vector_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "sparsha_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RETRIEVAL_POOL = max(10, int(os.getenv("RETRIEVAL_POOL", "40")))
MAX_CLASSIFY_CANDIDATES = max(1, int(os.getenv("MAX_CLASSIFY_CANDIDATES", "5")))
CLASSIFY_CONCURRENCY = max(1, int(os.getenv("CLASSIFY_CONCURRENCY", "5")))
MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.0"))
HTTP_RATE_LIMIT = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "12")))
RATE_WINDOW = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
GEMINI_RATE_LIMIT = max(1, int(os.getenv("GEMINI_RATE_LIMIT_CALLS", "12")))

DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
IAST_RE = re.compile(r"[āīūṛṝḷṅñṭḍṇśṣṃṁḥ]", re.I)
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|ok|okay|thanks?|thank you|namaste|good morning|good evening|"
    r"how are you|नमस्ते|नमस्कार|हेलो|कैसे हो)[?!. ]*$", re.I
)
CASUAL_SUPPORT_RE = re.compile(
    r"(?:^ya+r+$|\bbro\b|\bfriend\b|not feeling (?:good|well)|feeling (?:bad|low|sad|upset)|"
    r"i feel (?:bad|low|sad|unwell)|please help|मुझे अच्छा नहीं लग रहा|मन ठीक नहीं|यार)",
    re.I,
)
ROMAN_TO_DEVANAGARI = {
    "ajirna": "अजीर्ण", "amavata": "आमवात", "amlapitta": "अम्लपित्त",
    "arsha": "अर्श", "atisara": "अतिसार", "chardi": "छर्दि", "daha": "दाह",
    "grahani": "ग्रहणी", "gulma": "गुल्म", "hikka": "हिक्का", "jvara": "ज्वर",
    "kamala": "कामला", "kasa": "कास", "krimi": "कृमि", "kushtha": "कुष्ठ",
    "mutrakrichra": "मूत्रकृच्छ्र", "pandu": "पाण्डु", "prameha": "प्रमेह",
    "rajayakshma": "राजयक्ष्मा", "raktapitta": "रक्तपित्त", "shotha": "शोथ",
    "shula": "शूल", "svasa": "श्वास", "trishna": "तृष्णा", "udara": "उदर",
    "unmada": "उन्माद", "vatarakta": "वातरक्त", "visarpa": "विसर्प",
    "vrana": "व्रण",
    # Common BAMS typing variants (ASCII/non-IAST)
    "jwara": "ज्वर", "jwar": "ज्वर", "javara": "ज्वर",
    "madhumeha": "मधुमेह", "madhumeh": "मधुमेह",
    "shwasa": "श्वास", "shwas": "श्वास", "swasa": "श्वास",
    "kustha": "कुष्ठ", "kushta": "कुष्ठ", "kushtha": "कुष्ठ",
    "atisaar": "अतिसार", "atisara": "अतिसार",
    "mutrakricchra": "मूत्रकृच्छ्र", "mutrakrichhra": "मूत्रकृच्छ्र",
    "raktapita": "रक्तपित्त", "raktapitta": "रक्तपित्त",
    "sotha": "शोथ", "shoth": "शोथ", "shotha": "शोथ",
    "sula": "शूल", "shool": "शूल", "shula": "शूल",
}
KNOWN_SANSKRIT = set(ROMAN_TO_DEVANAGARI)
TERM_FALLBACKS = {"मधुमेह": ["प्रमेह"]}
DETERMINISTIC_OUTPUT = os.getenv("DETERMINISTIC_OUTPUT", "true").lower() in {"1", "true", "yes"}
POSITIVE_ACTION = re.compile(r"(?:हर|हरी|हृत्|घ्न|नाश|जित्|प्रणुत्|अपह|शमन)")
DANGER_ACTION = re.compile(r"(?:कर|कृत्|प्रद|वर्धन|कोपन|विदाह|मृत्यु|करोति)")
TERM_MAP_PATH = ROOT / "data" / "clinical_term_map.json"


def load_english_term_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not TERM_MAP_PATH.exists():
        return mapping
    try:
        payload = json.loads(TERM_MAP_PATH.read_text(encoding="utf-8"))
        for entry in payload.get("terms", []):
            devanagari = str(entry.get("sanskrit_devanagari", "")).strip()
            if not devanagari:
                continue
            names = [entry.get("english_name", ""), entry.get("sanskrit_roman", "")]
            names.extend(entry.get("english_aliases", []))
            for name in names:
                normalized = re.sub(r"[^a-z0-9]+", " ", str(name).casefold()).strip()
                if normalized:
                    mapping[normalized] = devanagari
    except Exception as exc:
        print(f"[startup] clinical term map could not be loaded: {exc}")
    return mapping


ENGLISH_TO_DEVANAGARI = load_english_term_map()

app = FastAPI(title="SPARSHA", version="2.0-fast")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SlidingLimiter:
    def __init__(self, limit: int, window: int) -> None:
        self.limit, self.window = limit, window
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self.lock:
            bucket = self.hits[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False, max(1, int(self.window - (now - bucket[0])) + 1)
            bucket.append(now)
            return True, 0


class ProviderLimiter:
    def __init__(self, limit: int, window: int = 60) -> None:
        self.limit, self.window = limit, window
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
                delay = max(0.05, self.window - (now - self.hits[0]) + 0.01)
            time.sleep(delay)


http_limiter = SlidingLimiter(HTTP_RATE_LIMIT, RATE_WINDOW)
gemini_limiter = ProviderLimiter(GEMINI_RATE_LIMIT)


@app.middleware("http")
async def limit_ai_requests(request: Request, call_next):
    if request.url.path in {"/ask", "/api/query"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        key = forwarded or (request.client.host if request.client else "unknown")
        allowed, retry = http_limiter.check(key)
        if not allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded."}, status_code=429,
                headers={"Retry-After": str(retry)},
            )
    return await call_next(request)


class AskRequest(BaseModel):
    # Supports both the current SSE frontend and the older UI request name.
    symptom: str | None = Field(default=None, max_length=500)
    message: str | None = Field(default=None, max_length=500)
    language: str = Field(default="sa", max_length=10)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class Candidate(BaseModel):
    doc_id: str
    distance: float
    document: str
    metadata: dict[str, Any]


class Verdict(BaseModel):
    status: Literal["SAFE", "DANGER", "NOT_FOUND"]
    reasoning: str
    candidate: Candidate | None = None


_resource_lock = threading.RLock()
_cache_lock = threading.Lock()
_collection = None
_gemini = None
_eligible_by_id: dict[str, Candidate] = {}
_normalize_cache: dict[str, list[str]] = {}
_classify_cache: dict[tuple[str, str], Verdict] = {}
_answer_cache: dict[str, tuple[str, str, str]] = {}


def canonical(text: str) -> str:
    return re.sub(r"[^a-zāīūṛṝḷṅñṭḍṇśṣṃṁḥ\u0900-\u097f]", "", text.casefold())


def extract_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
        if not match:
            raise ValueError("Model returned invalid JSON")
        return json.loads(match.group(1))


def get_collection():
    global _collection
    if _collection is None:
        with _resource_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                names = {item.name for item in client.list_collections()}
                if COLLECTION_NAME not in names:
                    raise RuntimeError(
                        f"Collection '{COLLECTION_NAME}' not found. Available: {sorted(names)}"
                    )
                embedding = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
                _collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding)
    return _collection


def get_gemini():
    global _gemini
    if _gemini is None:
        with _resource_lock:
            if _gemini is None:
                key = os.getenv("GEMINI_API_KEY", "").strip()
                if not key:
                    raise RuntimeError("GEMINI_API_KEY is missing from .env")
                _gemini = genai.Client(api_key=key)
    return _gemini


def load_eligible_records() -> None:
    global _eligible_by_id
    collection = get_collection()
    result = collection.get(
        where={"eligible_for_recommendation": True},
        include=["documents", "metadatas"],
    )
    loaded: dict[str, Candidate] = {}
    for doc_id, doc, meta in zip(
        result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
    ):
        loaded[doc_id] = Candidate(
            doc_id=doc_id, distance=999.0, document=doc or "", metadata=meta or {}
        )
    _eligible_by_id = loaded


@app.on_event("startup")
def warm_start() -> None:
    """Pay model/database loading cost at startup, not on the first question."""
    started = time.perf_counter()
    load_eligible_records()
    get_gemini()
    print(
        f"[startup] {len(_eligible_by_id)} eligible records loaded in "
        f"{time.perf_counter() - started:.2f}s"
    )


def is_sanskrit_query(query: str) -> bool:
    if DEVANAGARI_RE.search(query) or IAST_RE.search(query):
        return True
    tokens = [canonical(token) for token in TOKEN_RE.findall(query)]
    return bool(tokens) and len(tokens) <= 3 and all(token in KNOWN_SANSKRIT for token in tokens)


def map_english_query(query: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
    exact = ENGLISH_TO_DEVANAGARI.get(normalized)
    if exact:
        return exact
    # Longest phrase wins, so "heart disease" is preferred over "disease".
    matches: list[tuple[int, str]] = []
    padded = f" {normalized} "
    for phrase, devanagari in ENGLISH_TO_DEVANAGARI.items():
        if len(phrase) >= 4 and f" {phrase} " in padded:
            matches.append((len(phrase), devanagari))
    if not matches:
        return None
    matches.sort(reverse=True)
    best_length = matches[0][0]
    best_terms = {term for length, term in matches if length == best_length}
    return next(iter(best_terms)) if len(best_terms) == 1 else None


def normalize_query(query: str) -> list[str]:
    raw = query.strip()
    if is_sanskrit_query(raw):
        mapped = ROMAN_TO_DEVANAGARI.get(canonical(raw))
        terms = [raw]
        if mapped and mapped != raw:
            terms.append(mapped)
        terms.extend(TERM_FALLBACKS.get(mapped or raw, []))
        return list(dict.fromkeys(terms))
    verified_local = map_english_query(raw)
    if verified_local:
        return list(dict.fromkeys([raw, verified_local, *TERM_FALLBACKS.get(verified_local, [])]))
    cache_key = raw.casefold()
    with _cache_lock:
        if cache_key in _normalize_cache:
            return _normalize_cache[cache_key]

    prompt = f"""Normalize this English clinical query for retrieval from an Ayurvedic
Sanskrit Nighantu. Return ONLY a JSON array with the original key term and up to
five confident classical Sanskrit equivalents. Do not diagnose, prescribe, name
herbs, or guess uncertain equivalents.
Query: {raw}"""
    gemini_limiter.acquire()
    response = get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"
        ),
    )
    parsed = extract_json(response.text or "[]")
    if isinstance(parsed, dict):
        parsed = parsed.get("terms", [])
    terms = [raw]
    if isinstance(parsed, list):
        terms.extend(str(item).strip() for item in parsed if str(item).strip())
    terms = list(dict.fromkeys(terms))[:6]
    with _cache_lock:
        _normalize_cache[cache_key] = terms
    return terms


def retrieve_candidates(terms: list[str], raw_query: str) -> list[Candidate]:
    collection = get_collection()
    if not _eligible_by_id:
        return []
    merged: dict[str, Candidate] = {}
    pool_size = min(RETRIEVAL_POOL, len(_eligible_by_id))

    for term in terms:
        result = collection.query(
            query_texts=[term], n_results=pool_size,
            where={"eligible_for_recommendation": True},
            include=["documents", "metadatas", "distances"],
        )
        for doc_id, doc, meta, distance in zip(
            (result.get("ids") or [[]])[0],
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
        ):
            dist = float(distance)
            if dist > MAX_DISTANCE:
                continue
            old = merged.get(doc_id)
            if old is None or dist < old.distance:
                merged[doc_id] = Candidate(
                    doc_id=doc_id, distance=dist, document=doc or "", metadata=meta or {}
                )

    # Exact source-text matches are deterministic, nearly free, and improve recall.
    needles = {term.strip().casefold() for term in [raw_query, *terms] if len(term.strip()) >= 2}
    for doc_id, item in _eligible_by_id.items():
        if any(needle in item.document.casefold() for needle in needles):
            exact = item.model_copy(update={"distance": -1.0})
            local = local_explicit_verdict(raw_query, exact)
            if local and local.status == "SAFE":
                exact = exact.model_copy(update={"distance": -3.0})
            elif local and local.status == "DANGER":
                exact = exact.model_copy(update={"distance": -2.0})
            merged[doc_id] = exact

    return sorted(merged.values(), key=lambda item: item.distance)


def local_explicit_verdict(query: str, candidate: Candidate) -> Verdict | None:
    """Zero-LLM verdict only when Karma contains an explicit nearby action."""
    match = re.search(
        r"Raw karma Sanskrit:\s*(.*?)(?:\nEnglish disease:|\Z)",
        candidate.document,
        flags=re.S | re.I,
    )
    karma = match.group(1).strip() if match else ""
    if not karma:
        return None
    raw = query.strip()
    mapped = ROMAN_TO_DEVANAGARI.get(canonical(raw)) or map_english_query(raw)
    terms = [term for term in (raw if DEVANAGARI_RE.search(raw) else "", mapped) if term]
    terms.extend(TERM_FALLBACKS.get(mapped or raw, []))
    if not terms:
        return None

    found_positive = False
    found_danger = False
    for term in terms:
        for occurrence in re.finditer(re.escape(term), karma, flags=re.I):
            # Sanskrit Karma compounds normally put hara/ghna/kara directly after
            # the disease. A short window avoids borrowing an action from another disease.
            window = karma[occurrence.start() : occurrence.end() + 18]
            found_positive |= bool(POSITIVE_ACTION.search(window))
            found_danger |= bool(DANGER_ACTION.search(window))
    if found_positive and not found_danger:
        return Verdict(
            status="SAFE",
            reasoning="Raw Karma contains an explicit nearby disease-removing action.",
            candidate=candidate,
        )
    if found_danger and not found_positive:
        return Verdict(
            status="DANGER",
            reasoning="Raw Karma contains an explicit nearby aggravating/causative action.",
            candidate=candidate,
        )
    return None


def classify_one(query: str, candidate: Candidate) -> Verdict:
    key = (canonical(query), candidate.doc_id)
    with _cache_lock:
        cached = _classify_cache.get(key)
    if cached:
        return cached

    prompt = f"""You are a strict evidence gate for qualified BAMS consultants.
Evaluate ONLY this one record against the query. Return ONLY JSON with status and
reasoning.
SAFE = raw Karma/classical text explicitly indicates the queried condition.
DANGER = the record explicitly causes, aggravates, contraindicates, poisons, or
requires purification relevant to the query.
NOT_FOUND = absent, ambiguous, indirect, or inferred evidence.
Never infer missing facts and never classify a mere semantic similarity as SAFE.

Query: {query}
Record ID: {candidate.doc_id}
Record:
{candidate.document}"""
    gemini_limiter.acquire()
    response = get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"
        ),
    )
    parsed = extract_json(response.text or "{}")
    status = str(parsed.get("status", "NOT_FOUND")).upper()
    if status not in {"SAFE", "DANGER", "NOT_FOUND"}:
        status = "NOT_FOUND"
    verdict = Verdict(
        status=status,
        reasoning=str(parsed.get("reasoning", "No explicit evidence found.")).strip(),
        candidate=candidate,
    )
    with _cache_lock:
        _classify_cache[key] = verdict
    return verdict


def find_verdict_fast(query: str, candidates: list[Candidate]) -> Verdict:
    """Use strict local evidence first; parallel isolated LLM checks only if needed."""
    selected = candidates[:MAX_CLASSIFY_CANDIDATES]
    if not selected:
        return Verdict(status="NOT_FOUND", reasoning="I’m sorry you’re dealing with this. I could not find an eligible classical record for this concern, and I do not want to guess.")

    results: dict[str, Verdict] = {}
    unresolved: list[Candidate] = []
    for item in selected:
        local = local_explicit_verdict(query, item)
        if local:
            results[item.doc_id] = local
        else:
            unresolved.append(item)

    # An explicit local SAFE is stronger and faster than an embedding-only ambiguity.
    for item in selected:
        verdict = results.get(item.doc_id)
        if verdict and verdict.status == "SAFE":
            return verdict

    if unresolved:
        workers = min(CLASSIFY_CONCURRENCY, len(unresolved))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="safety") as executor:
            futures = {executor.submit(classify_one, query, item): item for item in unresolved}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results[item.doc_id] = future.result()
                except Exception as exc:
                    print(f"[classify] {item.doc_id} failed: {exc}")

    first_danger: Verdict | None = None
    for item in selected:  # deterministic distance order, not completion order
        verdict = results.get(item.doc_id)
        if not verdict:
            continue
        if verdict.status == "SAFE":
            return verdict
        if verdict.status == "DANGER" and first_danger is None:
            first_danger = verdict
    if first_danger:
        return first_danger
    return Verdict(
        status="NOT_FOUND",
        reasoning=f"I’m sorry you’re experiencing this. I could not verify an explicit indication in the {len(selected)} safety-checked records, so I’m withholding a recommendation rather than guessing.",
    )


def generate_grounded(query: str, candidate: Candidate) -> str:
    if DETERMINISTIC_OUTPUT:
        # The ingested document is already schema-formatted. Returning it directly
        # is faster and more faithful than asking an LLM to rewrite classical data.
        return candidate.document
    prompt = f"""Present this already-verified record concisely for a qualified BAMS
consultant. Use ONLY the record. Preserve Sanskrit exactly. Include Sanskrit name,
Varga, Rasa, Guna, Virya, Vipaka, Karma, source text, and verse ID when present.
Do not invent English names, botanical identity, dosage, formulation, or treatment.
Do not use markdown asterisks.

Query: {query}
Verified record:
{candidate.document}"""
    gemini_limiter.acquire()
    response = get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (response.text or "").strip()


def sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def done_event() -> str:
    return "data: [DONE]\n\n"


def friendly_followup(query: str, language: str) -> str:
    """One warm intake response; never diagnose or recommend treatment."""
    wants_hindi = language == "hi" or bool(DEVANAGARI_RE.search(query))
    fallback = (
        "मैं आपके साथ हूँ। कृपया आराम से बताइए—आपको सबसे अधिक क्या परेशानी हो रही है, "
        "यह कब शुरू हुई, और क्या दर्द, बुखार, खाँसी, साँस की परेशानी या कोई अन्य लक्षण है?"
        if wants_hindi else
        "I’m here with you. Please take your time—what is troubling you most, when did it "
        "start, and are you having pain, fever, cough, breathing difficulty, or another symptom?"
    )
    prompt = f"""You are SPARSHA's warm clinical-intake assistant. Respond with genuine empathy
without claiming to be a doctor. Do not diagnose, prescribe, name a remedy, or make promises.
Ask one short, natural follow-up question that helps the person describe the main symptom,
when it began, severity, and urgent warning signs. If they mention severe chest pain,
difficulty breathing, fainting, stroke signs, heavy bleeding, or self-harm, advise urgent
medical help. Respond entirely in {'Hindi' if wants_hindi else 'English'}.
User: {query}"""
    try:
        gemini_limiter.acquire()
        response = get_gemini().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=120),
        )
        text = (response.text or "").strip()
        return text or fallback
    except Exception as exc:
        print(f"[chat] Gemini follow-up failed; using safe fallback: {exc}")
        return fallback


def query_events(query: str, language: str = "en") -> Iterator[str]:
    started = time.perf_counter()
    cache_key = f"{language}:{query.strip().casefold()}"
    with _cache_lock:
        cached = _answer_cache.get(cache_key)
    if cached:
        status, reasoning, text = cached
        yield sse({"status": status, "reasoning": reasoning, "cached": True})
        if status in {"SAFE", "CHAT"} and text:
            yield sse({"text": text})
        yield done_event()
        return

    if GREETING_RE.fullmatch(query.strip()):
        hindi = language == "hi" or bool(DEVANAGARI_RE.search(query))
        message = (
            "नमस्ते! मैं आपकी बात ध्यान से सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? "
            "अपनी परेशानी आराम से बताइए; मैं उपलब्ध शास्त्रीय संदर्भ सावधानी से जाँचूँगा।"
            if hindi else
            "Namaste! I’m glad you reached out. How are you feeling today? Please describe "
            "your concern in your own words, and I’ll carefully check the available classical references."
        )
        yield sse({"status": "CHAT", "reasoning": "Friendly non-clinical conversation."})
        yield sse({"text": message})
        yield done_event()
        return

    # Vague distress needs a caring follow-up, not a failed herb search.
    if (
        CASUAL_SUPPORT_RE.search(query)
        and not is_sanskrit_query(query)
        and map_english_query(query) is None
    ):
        message = friendly_followup(query, language)
        with _cache_lock:
            _answer_cache[cache_key] = ("CHAT", "Supportive clinical-intake follow-up.", message)
        yield sse({"status": "CHAT", "reasoning": "Supportive clinical-intake follow-up."})
        yield sse({"text": message})
        yield done_event()
        return

    try:
        terms = normalize_query(query)
        candidates = retrieve_candidates(terms, query)
        verdict = find_verdict_fast(query, candidates)
        text = ""
        if verdict.status == "SAFE" and verdict.candidate:
            text = generate_grounded(query, verdict.candidate)
        with _cache_lock:
            _answer_cache[cache_key] = (verdict.status, verdict.reasoning, text)
        yield sse({
            "status": verdict.status,
            "reasoning": verdict.reasoning,
            "latency_seconds": round(time.perf_counter() - started, 3),
        })
        if verdict.status == "SAFE" and text:
            yield sse({"text": text})
    except Exception as exc:
        print(f"[query] failed: {exc}")
        yield sse({
            "status": "NOT_FOUND",
            "reasoning": "I’m sorry you’re dealing with this. I could not complete a verified classical lookup right now, so I will not guess or give an unsupported recommendation. Please try again shortly.",
        })
    yield done_event()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "total_records": get_collection().count(),
        "eligible_records": len(_eligible_by_id),
        "gemini_model": GEMINI_MODEL,
    }


@app.post("/ask")
def ask(body: AskRequest) -> StreamingResponse:
    text = (body.symptom or body.message or "").strip()
    if not text:
        return StreamingResponse(
            iter([sse({"status": "NOT_FOUND", "reasoning": "Query is empty."}), done_event()]),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        query_events(text, body.language), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query")
def query_post(body: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        query_events(body.query.strip()), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/query")
def query_get(q: str = Query(min_length=1, max_length=500)) -> StreamingResponse:
    return StreamingResponse(
        query_events(q.strip()), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


frontend = ROOT / "frontend"
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
