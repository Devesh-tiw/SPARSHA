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

PERSIST_PATH = "./chroma_ayurveda_db"
COLLECTION_NAME = "ayurvedic_herbs"

chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)

existing = {c.name: c for c in chroma_client.list_collections()}
print(f"[startup] Collections found at {PERSIST_PATH}: {list(existing.keys())}")

if COLLECTION_NAME not in existing:
    raise RuntimeError(
        f"Collection '{COLLECTION_NAME}' not found at {PERSIST_PATH}. "
        f"Available collections: {list(existing.keys())}. "
        f"Fix COLLECTION_NAME or PERSIST_PATH above to match your ingestion script."
    )

collection = chroma_client.get_collection(COLLECTION_NAME)
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


def retrieve_context(symptom: str) -> str:
    results = collection.query(query_texts=[symptom], n_results=5)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "No matching documents found."
    parts = [
        f"[Herb: {(meta or {}).get('herb_name', 'unknown')}]\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    return "\n\n---\n\n".join(parts)


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


async def stream_response(user_symptom: str, retrieved_context: str):
    try:
        result = await classify(user_symptom, retrieved_context)
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
    context = retrieve_context(req.symptom)
    return StreamingResponse(
        stream_response(req.symptom, context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if you're behind a proxy later
        },
    )


# uvicorn main:app --reload --port 8000