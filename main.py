import os
import json
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("LLAMA_API_KEY")
if not api_key:
    raise RuntimeError(
        "LLAMA_API_KEY is not set. Add it to a .env file next to main.py, "
        "or export it in your shell before running uvicorn."
    )

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

PERSIST_PATH = "./ayurveda_vector_db"
COLLECTION_NAME = "bhavprakash_collection"

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)

existing = {c.name: c for c in chroma_client.list_collections()}
print(f"[startup] Collections found at {PERSIST_PATH}: {list(existing.keys())}")

if COLLECTION_NAME not in existing:
    raise RuntimeError(
        f"Collection '{COLLECTION_NAME}' not found at {PERSIST_PATH}. "
        f"Available collections: {list(existing.keys())}. "
        f"Fix COLLECTION_NAME or PERSIST_PATH above to match your ingestion script."
    )

collection = chroma_client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
print(f"[startup] '{COLLECTION_NAME}' doc count: {collection.count()}")

if collection.count() == 0:
    print("[startup] WARNING: collection exists but is EMPTY. Ingestion never ran against this path/name.")


class AskRequest(BaseModel):
    symptom: str
    language: str = "en"


class Verdict(str, Enum):
    DANGER = "DANGER"
    NOT_FOUND = "NOT_FOUND"
    SAFE = "SAFE"


class Classification(BaseModel):
    verdict: Verdict
    matched_herb: str
    reasoning: str


NORMALIZE_PROMPT = """You expand patient symptom queries for an Ayurvedic vector search system. This runs on EVERY query regardless of which disease it is — do not special-case any particular condition.

Given a symptom in English or Hinglish, return a short comma-separated list of search terms including:
1. The original term as typed.
2. Its common English medical synonym(s), if different.
3. The classical Sanskrit/Ayurvedic disease term(s) most likely used for this condition in Charaka/Sushruta/Bhavaprakasha-style texts (e.g. diabetes -> Prameha/Madhumeha, fever -> Jwara, cough -> Kasa, asthma -> Shwasa, jaundice -> Kamala, arthritis -> Sandhivata/Amavata).

If you do not know a confident classical term for this symptom, just return the English synonyms — do not guess or invent a Sanskrit term.

Return ONLY the comma-separated term list, nothing else. No numbering, no explanation, no quotation marks.

Symptom: {raw_symptom}"""


async def normalize_query(raw_symptom: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": NORMALIZE_PROMPT.format(raw_symptom=raw_symptom)},
                {"role": "user", "content": "Expand."},
            ],
            temperature=0.0,
        )
        expanded = resp.choices[0].message.content.strip()
        return f"{raw_symptom}, {expanded}"
    except Exception:
        return raw_symptom


CLASSIFY_PROMPT = """You are an Ayurvedic Medical Verifier. Evaluate ONLY the single best-matching herb entry below against the patient symptom. Ignore contraindication/avoid-lists for OTHER diseases mentioned in the text — only judge this herb's relationship to THIS symptom.

Patient Symptom: "{user_symptom}"

Database Text:
---
{retrieved_context}
---

Rules:
- SAFE: the Karma (clinical action) section explicitly lists this symptom/disease as something the herb TREATS.
- DANGER: the text explicitly states this herb CAUSES, WORSENS, or is CONTRAINDICATED for this symptom.
- NOT_FOUND: the text is silent on this symptom for this herb.
- Do not infer from disease names appearing in unrelated context.

Return your answer via the classify function only. No other output."""

GENERATE_PROMPT = """You are an Ayurvedic formulary assistant. The herb below has already been verified SAFE for "{user_symptom}". Format its details. Do NOT use markdown asterisks. Do NOT include any tag words like [SAFE], [DANGER], or [NOT_FOUND] anywhere in your output.

Database Text:
---
{retrieved_context}
---

Format exactly:
Sanskrit Name: [Name]
English/Botanical Name: [Name]
Category (Varga): [Name]

Classical Properties:
- Rasa (Taste): [Extract]
- Guna (Qualities): [Extract]
- Virya (Potency): [Extract]
- Vipaka (Post-digestive): [Extract]

Clinical Action (Karma): [Extract]"""


def retrieve_candidates(symptom: str, candidates_per_term: int = 3, top_n: int = 5):
    """Query each normalized term separately (per fix #5), rank the merged
    candidates by actual embedding distance (not term/insertion order), and
    return the ranked list itself -- as (doc_id, distance, doc_text) tuples --
    rather than collapsing it into one string.

    Returning the whole ranked list (instead of joining just the top match
    into a single blob) is what lets the caller check herb #2, #3, ... when
    the closest embedding match doesn't happen to explicitly treat the
    symptom. This is dataset-wide and disease-agnostic: nothing here is
    keyed off any particular symptom or Sanskrit term, only distance scores.
    """
    terms = [t.strip() for t in symptom.split(",") if t.strip()]
    if not terms:
        terms = [symptom]

    best_by_id = {}  # doc_id -> (distance, doc_text)
    for term in terms:
        results = collection.query(
            query_texts=[term],
            n_results=candidates_per_term,
            include=["documents", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc_id, doc, dist in zip(ids, docs, dists):
            if doc_id not in best_by_id or dist < best_by_id[doc_id][0]:
                best_by_id[doc_id] = (dist, doc)

    ranked = sorted(best_by_id.items(), key=lambda kv: kv[1][0])[:top_n]

    preview = ", ".join(f"{doc_id}(d={dist:.3f})" for doc_id, (dist, _) in ranked)
    print(f"[retrieve] ranked candidates: {preview or 'none'}")

    return [(doc_id, dist, doc) for doc_id, (dist, doc) in ranked]


async def classify(user_symptom: str, retrieved_context: str) -> Classification:
    resp = await client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT.format(
                user_symptom=user_symptom, retrieved_context=retrieved_context)},
            {"role": "user", "content": "Classify."},
        ],
        temperature=0.0,
        response_format={"type": "json_schema", "json_schema": {
            "name": "classification",
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["DANGER", "NOT_FOUND", "SAFE"]},
                    "matched_herb": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["verdict", "matched_herb", "reasoning"],
                "additionalProperties": False,
            },
        }},
    )
    return Classification.model_validate_json(resp.choices[0].message.content)


async def find_verdict(user_symptom: str, candidates: list):
    """Walk the ranked candidates one herb at a time -- the closest embedding
    match first -- and stop at the first one explicitly confirmed SAFE.

    Each herb is still judged strictly on its own text alone (never blended
    with others), which is what actually fixed the original mis-classification.
    But instead of giving up after herb #1, we keep going down the ranked list
    if herb #1 turns out to be silent on this symptom -- since with ~1,110
    herbs, the single nearest embedding match often isn't the only one worth
    checking. If a closer herb is explicitly DANGER for this symptom, that's
    surfaced (safety-first) unless a later, still-close candidate is SAFE.
    No disease name or herb name is special-cased here; this applies to every
    query the same way.
    """
    first_danger = None
    for doc_id, dist, doc in candidates:
        try:
            result = await classify(user_symptom, doc)
        except Exception as e:
            print(f"[classify] candidate={doc_id} errored: {e}")
            continue
        print(f"[classify] candidate={doc_id} (d={dist:.3f}) -> {result.verdict}")
        if result.verdict == Verdict.SAFE:
            return result, doc
        if result.verdict == Verdict.DANGER and first_danger is None:
            first_danger = (result, doc)

    if first_danger:
        return first_danger
    return Classification(
        verdict=Verdict.NOT_FOUND,
        matched_herb="",
        reasoning="None of the closest classical matches explicitly treat this symptom.",
    ), None


async def stream_response(user_symptom: str, candidates: list):
    if not candidates:
        yield f"data: {json.dumps({'status': 'NOT_FOUND', 'reasoning': 'No matching documents found.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        result, matched_doc = await find_verdict(user_symptom, candidates)
    except Exception as e:
        yield f"data: {json.dumps({'status': 'NOT_FOUND', 'reasoning': f'Classifier error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if result.verdict == Verdict.DANGER:
        yield f"data: {json.dumps({'status': 'DANGER', 'reasoning': result.reasoning})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if result.verdict == Verdict.NOT_FOUND:
        yield f"data: {json.dumps({'status': 'NOT_FOUND', 'reasoning': result.reasoning})}\n\n"
        yield "data: [DONE]\n\n"
        return

    retrieved_context = matched_doc
    yield f"data: {json.dumps({'status': 'SAFE'})}\n\n"

    response_stream = await client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": GENERATE_PROMPT.format(
                user_symptom=user_symptom, retrieved_context=retrieved_context)},
            {"role": "user", "content": f"Format the remedy for: {user_symptom}"},
        ],
        temperature=0.0,
        stream=True,
    )

    async for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {json.dumps({'text': chunk.choices[0].delta.content})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/ask")
async def ask(req: AskRequest):
    normalized_symptom = await normalize_query(req.symptom)
    print(f"[normalize] '{req.symptom}' -> '{normalized_symptom}'")
    candidates = retrieve_candidates(normalized_symptom)
    return StreamingResponse(
        stream_response(req.symptom, candidates),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )