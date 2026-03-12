# ==========================================================
# CELO NIS AI AGENT (PRODUCTION VERSION)
# PART 1 — CORE INFRASTRUCTURE
# ==========================================================

import streamlit as st
import os
import re
import json
import pickle
import logging
import time
import requests
import numpy as np
import faiss

from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="Celonis AI Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# GLOBAL CONFIG
# ==========================================================

MODEL_FAST = "llama-3.1-8b-instant"
MODEL_REASON = "mixtral-8x7b-32768"

TOP_K_RETRIEVAL = 5
MAX_CONTEXT_CHARS = 6000

DATA_DIR = "./"

DOC_INDEX_FILE = "pql_faiss.index"
DOC_META_FILE = "pql_metadata.pkl"

QA_INDEX_FILE = "pql_knowledge.index"
QA_META_FILE = "pql_knowledge.pkl"

DYNAMIC_INDEX_FILE = "dynamic_docs.index"
DYNAMIC_META_FILE = "dynamic_docs.pkl"

LEARN_FILE = "learned_examples.pkl"

LOG_FILE = "agent.log"

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_event(msg):
    logging.info(msg)

# ==========================================================
# GROQ CLIENT
# ==========================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    rerank_model = CrossEncoder("BAAI/bge-reranker-base")

    return embed_model, rerank_model

EMBED_MODEL, RERANK_MODEL = load_models()

# ==========================================================
# CACHE SYSTEM
# ==========================================================

@st.cache_data(ttl=3600)
def cached_embed(text):

    emb = EMBED_MODEL.encode([text])

    return np.array(emb).astype("float32")

# ==========================================================
# FILE UTILITIES
# ==========================================================

def safe_pickle_load(path):

    if not os.path.exists(path):
        return None

    with open(path,"rb") as f:
        return pickle.load(f)

def safe_pickle_save(path,obj):

    with open(path,"wb") as f:
        pickle.dump(obj,f)

# ==========================================================
# VECTOR STORE LOADING
# ==========================================================

def load_vector_store(index_path,meta_path):

    if os.path.exists(index_path):

        index = faiss.read_index(index_path)

        meta = safe_pickle_load(meta_path)

        if meta is None:
            meta = []

    else:

        index = faiss.IndexFlatL2(384)
        meta = []

    return index,meta

# ==========================================================
# LOAD STATIC KNOWLEDGE
# ==========================================================

docs_index, docs_meta = load_vector_store(
    DOC_INDEX_FILE,
    DOC_META_FILE
)

qa_index, qa_meta = load_vector_store(
    QA_INDEX_FILE,
    QA_META_FILE
)

# ==========================================================
# LOAD DYNAMIC KNOWLEDGE
# ==========================================================

dynamic_index, dynamic_meta = load_vector_store(
    DYNAMIC_INDEX_FILE,
    DYNAMIC_META_FILE
)

# ==========================================================
# BM25 INDEX
# ==========================================================

if docs_meta:

    docs_tokens = [
        re.findall(r"\w+", item["text"].lower())
        for item in docs_meta
    ]

    docs_bm25 = BM25Okapi(docs_tokens)

else:

    docs_bm25 = None

# ==========================================================
# SELF LEARNING MEMORY
# ==========================================================

if os.path.exists(LEARN_FILE):

    learned_examples = safe_pickle_load(LEARN_FILE)

else:

    learned_examples = []

def store_learning(question,answer):

    if len(answer) < 200:
        return

    learned_examples.append({
        "q":question,
        "a":answer
    })

    safe_pickle_save(LEARN_FILE,learned_examples)

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def normalize_text(t):

    return re.sub(r"\s+"," ",t).strip()

def chunk_text(text,size=500):

    chunks = []

    for i in range(0,len(text),size):

        chunks.append(text[i:i+size])

    return chunks


# ==========================================================
# PART 2 — KNOWLEDGE RETRIEVAL ENGINE
# ==========================================================

# ==========================================================
# VECTOR SEARCH
# ==========================================================

def vector_search(query, index, metadata, top_k=TOP_K_RETRIEVAL):

    if len(metadata) == 0:
        return []

    query_emb = cached_embed(query)

    D, I = index.search(query_emb, top_k)

    results = []

    for idx in I[0]:

        if idx < len(metadata):

            item = metadata[idx]

            if isinstance(item, dict):
                results.append(item.get("text",""))
            else:
                results.append(item)

    return results


# ==========================================================
# BM25 SEARCH
# ==========================================================

def bm25_search(query, top_k=TOP_K_RETRIEVAL):

    if docs_bm25 is None:
        return []

    tokens = re.findall(r"\w+", query.lower())

    scores = docs_bm25.get_scores(tokens)

    ranked = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in ranked:

        if scores[i] > 0:

            results.append(docs_meta[i]["text"])

    return results


# ==========================================================
# DEDUPLICATION
# ==========================================================

def deduplicate(docs):

    seen = set()
    unique = []

    for d in docs:

        key = d[:100]

        if key not in seen:

            seen.add(key)
            unique.append(d)

    return unique


# ==========================================================
# CROSS ENCODER RERANKING
# ==========================================================

def rerank_results(query, docs, top_k=4):

    if not docs:
        return []

    pairs = [[query, doc] for doc in docs]

    scores = RERANK_MODEL.predict(pairs)

    ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

    return ranked[:top_k]


# ==========================================================
# LEARNED MEMORY SEARCH
# ==========================================================

def search_learned_memory(query):

    if not learned_examples:
        return []

    tokens = set(re.findall(r"\w+", query.lower()))

    results = []

    for item in learned_examples:

        q_tokens = set(re.findall(r"\w+", item["q"].lower()))

        overlap = len(tokens & q_tokens)

        if overlap > 2:

            results.append(item["a"])

    return results[:2]


# ==========================================================
# CELO NIS DOC SCRAPER
# ==========================================================

def scrape_celonis_docs(query):

    try:

        search_url = f"https://docs.celonis.com/en/search.html?q={query}"

        r = requests.get(
            search_url,
            timeout=10,
            headers={"User-Agent":"Mozilla/5.0"}
        )

        soup = BeautifulSoup(r.text, "html.parser")

        links = []

        for a in soup.find_all("a", href=True):

            href = a["href"]

            if "/en/" in href and ".html" in href:

                links.append("https://docs.celonis.com" + href)

        if not links:
            return None

        page = requests.get(links[0], timeout=10)

        page_soup = BeautifulSoup(page.text, "html.parser")

        paragraphs = page_soup.find_all("p")

        text = "\n".join(p.get_text() for p in paragraphs)

        return text[:5000]

    except Exception as e:

        log_event(f"Doc scrape failed: {e}")

        return None


# ==========================================================
# ADD DYNAMIC DOCUMENTS
# ==========================================================

def add_dynamic_docs(text):

    chunks = chunk_text(text)

    embeddings = EMBED_MODEL.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dynamic_index.add(embeddings)

    dynamic_meta.extend(chunks)

    faiss.write_index(dynamic_index, DYNAMIC_INDEX_FILE)

    safe_pickle_save(DYNAMIC_META_FILE, dynamic_meta)

    log_event("Dynamic docs added to FAISS store")


# ==========================================================
# SEARCH DYNAMIC STORE
# ==========================================================

def search_dynamic_store(query):

    if len(dynamic_meta) == 0:
        return []

    emb = cached_embed(query)

    D, I = dynamic_index.search(emb, 3)

    docs = []

    for idx in I[0]:

        if idx < len(dynamic_meta):

            docs.append(dynamic_meta[idx])

    return docs


# ==========================================================
# HYBRID RETRIEVAL PIPELINE
# ==========================================================

def hybrid_retrieve(query):

    docs = []

    # semantic search docs
    docs += vector_search(query, docs_index, docs_meta)

    # semantic search examples
    docs += vector_search(query, qa_index, qa_meta)

    # dynamic knowledge
    docs += search_dynamic_store(query)

    # keyword search
    docs += bm25_search(query)

    # learned memory
    docs += search_learned_memory(query)

    docs = deduplicate(docs)

    docs = rerank_results(query, docs)

    # fallback to web scraping
    if not docs:

        scraped = scrape_celonis_docs(query)

        if scraped:

            add_dynamic_docs(scraped)

            docs = [scraped]

    context = "\n\n".join(docs)

    return context[:MAX_CONTEXT_CHARS]


# ==========================================================
# FUNCTION DETECTION
# ==========================================================

PQL_FUNCTIONS = [
    "PU_FIRST","PU_LAST","PU_AVG","PU_SUM","PU_COUNT",
    "PU_MIN","PU_MAX","PU_COUNT_DISTINCT",
    "DATEDIFF","RUNNING_SUM","COUNT_TABLE",
    "MOVING_AVG","REMAP","AVG","MEDIAN",
    "SOURCE","TARGET","RUNNING_TOTAL"
]

def detect_function(query):

    q = query.upper()

    for f in PQL_FUNCTIONS:

        if re.search(rf"\b{f}\b", q):

            return f

    return None


# ==========================================================
# FUNCTION ROUTING
# ==========================================================

def route_function_doc(function_name):

    normalized = function_name.lower().replace("_","-")

    for item in docs_meta:

        url = item.get("url","").lower()

        if normalized in url:

            return item["text"]

    return None


# ==========================================================
# RETRIEVE CONTEXT
# ==========================================================

def retrieve_context(query):

    func = detect_function(query)

    if func:

        routed = route_function_doc(func)

        if routed:

            return routed

    context = hybrid_retrieve(query)

    return context


# ==========================================================
# PART 3 — AGENT SYSTEM
# ==========================================================

# ==========================================================
# INTENT AGENT
# ==========================================================

def detect_intent(prompt):

    p = prompt.lower()

    if any(x in p for x in ["write","generate","build","create","pql"]):
        return "pql_generation"

    if any(x in p for x in ["what is","explain","definition"]):
        return "explanation"

    if any(x in p for x in [
        "cycle time","throughput","rework",
        "bottleneck","process","variant"
    ]):
        return "business_problem"

    return "general"


# ==========================================================
# QUERY REWRITE AGENT
# ==========================================================

def rewrite_query(prompt):

    try:

        r = client.chat.completions.create(
            model = MODEL_FAST,
            messages=[{
                "role":"user",
                "content":f"""
Rewrite the following question using Celonis PQL terminology.

Question:
{prompt}

Return only the rewritten search query.
"""
            }],
            temperature=0,
            max_tokens=100
        )

        return r.choices[0].message.content.strip()

    except:

        return prompt


# ==========================================================
# MULTI QUERY GENERATION
# ==========================================================

def generate_multi_queries(query):

    try:

        r = client.chat.completions.create(
            model = MODEL_FAST,
            messages=[{
                "role":"user",
                "content":f"""
Generate 3 alternative Celonis documentation search queries.

Query:
{query}

Return each query on a new line.
"""
            }],
            temperature=0.3,
            max_tokens=120
        )

        lines = r.choices[0].message.content.split("\n")

        return [l.strip() for l in lines if l.strip()]

    except:

        return []


# ==========================================================
# PQL GENERATOR AGENT
# ==========================================================

def generate_pql(prompt, context):

    system_prompt = """
You are a senior Celonis Process Mining consultant.

Rules:
- Only generate valid PQL
- Never generate SQL
- Quote tables like "Table"."Column"
- Use official Celonis functions
"""

    user_prompt = f"""
User question:
{prompt}

Reference context:
{context}

Write the PQL query and explain it.
"""

    r = client.chat.completions.create(

        model = MODEL_REASON,

        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],

        temperature=0.2,
        max_tokens=800
    )

    return r.choices[0].message.content


# ==========================================================
# EXPLANATION AGENT
# ==========================================================

def explain_concept(prompt, context):

    r = client.chat.completions.create(

        model = MODEL_FAST,

        messages=[
            {"role":"system","content":
             "Explain Celonis concepts clearly."},
            {"role":"user","content":
             prompt + "\n\n" + context}
        ],

        temperature=0.2
    )

    return r.choices[0].message.content


# ==========================================================
# VALIDATION AGENT
# ==========================================================

def validate_pql(answer):

    sql_keywords = ["SELECT","FROM","WHERE","JOIN","GROUP BY"]

    for k in sql_keywords:

        if k in answer.upper():

            return False

    return True


# ==========================================================
# TOOL ROUTER
# ==========================================================

def route_tools(intent):

    if intent == "pql_generation":

        return [
            "retrieve_docs",
            "generate_pql"
        ]

    if intent == "business_problem":

        return [
            "retrieve_docs",
            "generate_pql"
        ]

    if intent == "explanation":

        return [
            "retrieve_docs",
            "explain"
        ]

    return [
        "retrieve_docs",
        "explain"
    ]


# ==========================================================
# QUERY PLANNER AGENT
# ==========================================================

def planner(prompt):

    intent = detect_intent(prompt)

    tools = route_tools(intent)

    log_event(f"Intent detected: {intent}")
    log_event(f"Tools selected: {tools}")

    return intent, tools


# ==========================================================
# EXECUTION ENGINE
# ==========================================================

def run_agent(prompt):

    intent, tools = planner(prompt)

    rewritten = rewrite_query(prompt)

    queries = [rewritten] + generate_multi_queries(rewritten)

    context = ""

    if "retrieve_docs" in tools:

        all_context = []

        for q in queries:

            all_context.append(
                retrieve_context(q)
            )

        context = "\n\n".join(all_context)

    if "generate_pql" in tools:

        answer = generate_pql(prompt, context)

    else:

        answer = explain_concept(prompt, context)

    if not validate_pql(answer):

        answer = "⚠️ Generated SQL instead of valid PQL."

    return answer


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🧠 Celonis Process Mining AI Copilot")

st.markdown(
"""
Ask questions about **Celonis**, **Process Mining**, or **PQL queries**.
"""
)

# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================================
# CHAT HISTORY
# ==========================================================

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])


# ==========================================================
# USER INPUT
# ==========================================================

prompt = st.chat_input("Ask a Celonis question...")


if prompt:

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing process knowledge..."):

            try:

                answer = run_agent(prompt)

            except Exception as e:

                answer = f"⚠️ Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })

    store_learning(prompt, answer)
