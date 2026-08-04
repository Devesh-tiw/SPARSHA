from __future__ import annotations

import os
import sqlite3
from enum import Enum
from typing import List, Optional, TypedDict
from dotenv import load_dotenv

load_dotenv()  

from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# 1. Models & Embeddings
# ---------------------------------------------------------------------------
GENERATOR_LLM = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["LLAMA_API_KEY"], 
    temperature=0.3,
)

VERIFIER_LLM = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct", 
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["LLAMA_API_KEY"],
    temperature=0.0, 
)

NORMALIZER_LLM = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["LLAMA_API_KEY"],
    temperature=0.0,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="ayurvedic_herbs",
    embedding_function=embeddings,
    persist_directory="./chroma_ayurveda_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ---------------------------------------------------------------------------
# 2. Caching & Tools
# ---------------------------------------------------------------------------
class SQLiteSearchCache:
    """Persistent disk-based SQLite cache for web search queries."""
    def __init__(self, db_path="verification_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tavily_cache (
                    query TEXT PRIMARY KEY,
                    evidence TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get(self, query: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT evidence FROM tavily_cache WHERE query = ?", (query,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def set(self, query: str, evidence: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tavily_cache (query, evidence) VALUES (?, ?)",
                (query, evidence)
            )

search_cache = SQLiteSearchCache()

TRUSTED_DOMAINS = [
    "ayush.gov.in",
    "nccih.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "ccras.nic.in", 
]

tavily_tool = TavilySearchResults(
    max_results=5,
    include_domains=TRUSTED_DOMAINS,
)


# ---------------------------------------------------------------------------
# 3. State & Pydantic Models
# ---------------------------------------------------------------------------
SAFE_FAILURE_MESSAGE = (
    "I could not verify a safe and clinically appropriate Ayurvedic remedy "
    "for this query with sufficient confidence. Rather than risk providing "
    "inaccurate or contraindicated guidance, I'm withholding a recommendation. "
    "Please consult a registered Ayurvedic physician (BAMS) for this concern."
)

MAX_VERIFICATION_RETRIES = 2 

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"

class ClinicalVerification(BaseModel):
    verdict: Verdict = Field(
        description="PASS only if safe. FAIL if contraindicated/cause. UNCERTAIN if ambiguous."
    )
    is_contraindicated: bool = Field(description="True if contraindicated for symptom.")
    hallucination_detected: bool = Field(description="True if property invented.")
    cause_vs_cure_conflict: bool = Field(description="True if herb causes rather than cures.")
    reasoning: str = Field(description="Clinical reasoning citing Sanskrit property.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1.")

class PipelineState(TypedDict, total=False):
    raw_query: str
    normalized_query: str
    retrieved_docs: List[Document]
    context_text: str
    draft_answer: str
    verification: ClinicalVerification
    external_evidence: Optional[str]
    retry_count: int
    final_answer: str
    verdict_log: List[str]


# ---------------------------------------------------------------------------
# 4. Graph Nodes
# ---------------------------------------------------------------------------
normalizer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You normalize patient queries for an Ayurvedic vector search system. "
     "Fix typos, translate any Hinglish or Hindi terms to English, and expand "
     "colloquial symptom descriptions into precise clinical/Ayurvedic terms. "
     "Return only the normalized query text, nothing else."),
    ("human", "{raw_query}"),
])
normalizer_chain = normalizer_prompt | NORMALIZER_LLM | StrOutputParser()

def normalize_query_node(state: PipelineState) -> PipelineState:
    normalized = normalizer_chain.invoke({"raw_query": state["raw_query"]})
    return {"normalized_query": normalized.strip()}

def retrieve_node(state: PipelineState) -> PipelineState:
    docs = retriever.invoke(state["normalized_query"])
    context_text = "\n\n---\n\n".join(
        f"[Herb: {d.metadata.get('herb_name', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    return {"retrieved_docs": docs, "context_text": context_text}

generator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an Ayurvedic knowledge assistant. Answer ONLY using the "
     "retrieved context below. Never use outside knowledge. Cite properties used.\n\n"
     "Retrieved context:\n{context}"),
    ("human", "{query}"),
])
generator_chain = generator_prompt | GENERATOR_LLM | StrOutputParser()

def generate_draft_node(state: PipelineState) -> PipelineState:
    draft = generator_chain.invoke({
        "context": state["context_text"],
        "query": state["normalized_query"],
    })
    return {"draft_answer": draft}

VERIFIER_SYSTEM_PROMPT = """You are a strict Ayurvedic Medical Board reviewer. \
You did NOT write the draft answer below — your only job is to find reasons \
to REJECT it. You are adversarial toward the draft, not cooperative with it.

You must check, in order:
1. CAUSE-VS-CURE CHECK: Does the retrieved context describe this herb as a cause or aggravating factor?
2. HALLUCINATION CHECK: Are all claims in the draft supported by the context?
3. CONTRAINDICATION CHECK: Is the herb contraindicated by its classical properties?
4. If you are not fully certain after checks 1-3, you MUST return UNCERTAIN.

Retrieved context:
{context}

Original patient query: {query}

Draft answer to review:
{draft}
"""
verifier_prompt = ChatPromptTemplate.from_messages([("system", VERIFIER_SYSTEM_PROMPT)])
verifier_chain = verifier_prompt | VERIFIER_LLM.with_structured_output(ClinicalVerification)

def verify_node(state: PipelineState) -> PipelineState:
    context = state["context_text"]
    if state.get("external_evidence"):
        context += f"\n\n---\nEXTERNAL VERIFICATION EVIDENCE:\n{state['external_evidence']}"

    result: ClinicalVerification = verifier_chain.invoke({
        "context": context,
        "query": state["normalized_query"],
        "draft": state["draft_answer"],
    })

    log = state.get("verdict_log", [])
    log.append(f"[retry={state.get('retry_count', 0)}] {result.verdict}: {result.reasoning}")
    return {"verification": result, "verdict_log": log}

def external_verify_node(state: PipelineState) -> PipelineState:
    herb_names = ", ".join(d.metadata.get("herb_name", "") for d in state.get("retrieved_docs", []))
    search_query = f"{herb_names} contraindications indications {state['normalized_query']} classical Ayurvedic evidence"

    cached_evidence = search_cache.get(search_query)
    if cached_evidence:
        print("⚡ [CACHE HIT] Served from SQLite.")
        return {"external_evidence": cached_evidence, "retry_count": state.get("retry_count", 0) + 1}

    print("🌐 [CACHE MISS] Fetching from Tavily API...")
    try:
        results = tavily_tool.invoke({"query": search_query})
        evidence = "\n\n".join(f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}" for r in results)
        search_cache.set(search_query, evidence)
    except Exception as e:
        evidence = f"[External verification unavailable: {e}]"

    return {"external_evidence": evidence, "retry_count": state.get("retry_count", 0) + 1}

def route_after_verification(state: PipelineState) -> str:
    v = state["verification"]
    retry_count = state.get("retry_count", 0)
    if v.verdict == Verdict.PASS and v.confidence >= 0.75: return "finalize"
    if v.verdict == Verdict.FAIL: return "safe_failure"
    if retry_count < MAX_VERIFICATION_RETRIES: return "external_verify"
    return "safe_failure"

def finalize_node(state: PipelineState) -> PipelineState:
    disclaimer = "\n\n---\n*This response has passed automated clinical verification.*"
    return {"final_answer": state["draft_answer"] + disclaimer}

def safe_failure_node(state: PipelineState) -> PipelineState:
    return {"final_answer": SAFE_FAILURE_MESSAGE}


# ---------------------------------------------------------------------------
# 5. Build Graph
# ---------------------------------------------------------------------------
def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("normalize_query", normalize_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_draft", generate_draft_node)
    graph.add_node("verify", verify_node)
    graph.add_node("external_verify", external_verify_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("safe_failure", safe_failure_node)

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "generate_draft")
    graph.add_edge("generate_draft", "verify")

    graph.add_conditional_edges(
        "verify", route_after_verification,
        {"finalize": "finalize", "safe_failure": "safe_failure", "external_verify": "external_verify"},
    )
    graph.add_edge("external_verify", "verify")
    graph.add_edge("finalize", END)
    graph.add_edge("safe_failure", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_pipeline()

    # Try running a test query
    result = pipeline.invoke({
        "raw_query": "mujhe sugar ki problem hai, koi ayurvedic ilaj batao",
        "retry_count": 0,
        "verdict_log": [],
    })

    print("NORMALIZED QUERY:", result["normalized_query"])
    print("\nFINAL ANSWER:\n", result["final_answer"])
    print("\n--- AUDIT TRAIL ---")
    for entry in result["verdict_log"]:
        print(entry)