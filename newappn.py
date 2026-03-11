import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64
from rank_bm25 import BM25Okapi

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# BACKGROUND SETUP
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        page_bg = f"""
        <style>

        /* Full App Background */
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{
            background-color: transparent !important;
        }}

        section.main > div {{
            background-color: transparent !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}

        .stChatMessage {{
            background-color: rgba(20, 20, 30, 0.85);
            border-radius: 12px;
            padding: 12px;
        }}

        div[data-testid="stChatInput"] {{
            background-color: rgba(20, 20, 30, 0.95);
            border-radius: 12px;
            padding: 8px;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 15, 25, 0.95);
        }}

        h1, h2, h3, p, div, span {{
            color: white !important;
        }}

        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

STRICT RULES:

1. For general Process Mining questions:
   - Provide structured explanation.
   - Use simple real-world examples.

2. For Celonis / PQL questions:
   - STRICTLY use provided documentation context AND example queries.
   - Preserve official syntax EXACTLY.
   - Do NOT convert PQL into SQL.
   - Do NOT simplify syntax.
   - Do NOT invent examples outside of provided context.
   - If not found in documentation, say:
     "Not found in official Celonis documentation."

3. Only generate new PQL queries if explicitly requested.
4. Never use SQL keywords like SELECT, FROM, WHERE, JOIN.
5. When example PQL queries are provided in context, reference them to guide your answer.
6. Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# LOAD BOTH VECTOR STORES + BM25
# =====================================

@st.cache_resource
def load_all_stores():
    # --- Official Celonis Docs ---
    docs_index = faiss.read_index("pql_faiss.index")
    with open("pql_metadata.pkl", "rb") as f:
        docs_metadata = pickle.load(f)      # list of dicts: {url, title, text}

    # --- PQL Q&A Examples ---
    qa_index = faiss.read_index("pql_knowledge.index")
    with open("pql_knowledge.pkl", "rb") as f:
        qa_metadata = pickle.load(f)        # list of strings

    # --- BM25 on Docs ---
    docs_corpus = [item["text"].lower().split() for item in docs_metadata]
    docs_bm25 = BM25Okapi(docs_corpus)

    # --- BM25 on Q&A ---
    qa_corpus = [entry.lower().split() for entry in qa_metadata]
    qa_bm25 = BM25Okapi(qa_corpus)

    return docs_index, docs_metadata, docs_bm25, qa_index, qa_metadata, qa_bm25

docs_index, docs_metadata, docs_bm25, qa_index, qa_metadata, qa_bm25 = load_all_stores()

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_rewritten" not in st.session_state:
    st.session_state.last_rewritten = ""

# =====================================
# QUERY DETECTION
# =====================================

def is_celonis_query(prompt):
    keywords = [
        "celonis", "pql", "pu_", "datediff", "remap",
        "process query language", "pull-up", "running_sum",
        "throughput", "event log", "case table", "activity table",
        "conformance", "variant", "filter", "process mining",
        "count_table", "avg", "median", "moving_avg", "process",
        "avg_process", "source", "target", "rework", "loop",
        "calc", "calculate", "query", "function", "column",
        "aggregation", "metric", "kpi", "timestamp", "duration",
        "case", "activity", "edge", "node", "frequency"
    ]
    return any(word in prompt.lower() for word in keywords)

# =====================================
# STEP 1 — QUERY REWRITING WITH LLM
# =====================================

def rewrite_query(prompt):
    """Rewrites user question into precise PQL/Celonis terminology before searching."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a Celonis PQL expert assistant.
Rewrite the user's question into precise PQL and process mining terminology
so it can be used to search a Celonis documentation database.

Rules:
- Use official PQL function names where applicable (e.g., PU_AVG, DATEDIFF, COUNT_TABLE, RUNNING_SUM)
- Use Celonis-specific terms (e.g., case table, activity table, event log, pull-up functions)
- Return ONLY the rewritten search query — no explanation, no extra text
- Keep it concise (1-2 sentences max)

User Question: {prompt}

Rewritten Search Query:"""
                }
            ],
            max_tokens=100,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return prompt  # Fallback to original if rewriting fails

# =====================================
# STEP 2 — EXACT FUNCTION ROUTING
# =====================================

def exact_function_match(query):
    """Directly routes known PQL function names to their exact doc entry."""
    query_upper = query.upper()
    tokens = re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', query_upper)

    for token in tokens:
        for item in docs_metadata:
            url = item.get("url", "").lower()
            if token.lower().replace("_", "-") in url or token.lower() in url:
                return item["text"]
    return None

# =====================================
# STEP 3 — HYBRID SEARCH (SEMANTIC + BM25)
# =====================================

def hybrid_search(query, top_k=3):
    """
    Runs semantic (FAISS cosine) + keyword (BM25) search on both stores.
    Merges results with weighted scoring: 60% semantic + 40% BM25.
    """
    query_embedding = EMBED_MODEL.encode([query])
    query_np = np.array(query_embedding)
    query_tokens = query.lower().split()

    # ---- DOCS: Semantic ----
    D_docs, I_docs = docs_index.search(query_np, top_k * 2)
    semantic_doc_scores = {
        int(idx): 1 / (1 + float(dist))
        for idx, dist in zip(I_docs[0], D_docs[0])
        if idx < len(docs_metadata)
    }

    # ---- DOCS: BM25 ----
    bm25_doc_raw = docs_bm25.get_scores(query_tokens)
    bm25_doc_top_idx = np.argsort(bm25_doc_raw)[::-1][:top_k * 2]
    max_bm25_doc = bm25_doc_raw[bm25_doc_top_idx[0]] if bm25_doc_raw[bm25_doc_top_idx[0]] > 0 else 1
    bm25_doc_scores = {
        int(i): bm25_doc_raw[i] / max_bm25_doc
        for i in bm25_doc_top_idx
        if bm25_doc_raw[i] > 0
    }

    # ---- DOCS: Merge ----
    all_doc_idx = set(semantic_doc_scores) | set(bm25_doc_scores)
    merged_doc = {
        i: (semantic_doc_scores.get(i, 0) * 0.6) + (bm25_doc_scores.get(i, 0) * 0.4)
        for i in all_doc_idx
    }
    top_docs = sorted(merged_doc, key=merged_doc.get, reverse=True)[:top_k]
    doc_results = "\n\n".join(docs_metadata[i]["text"] for i in top_docs)

    # ---- QA: Semantic ----
    D_qa, I_qa = qa_index.search(query_np, top_k * 2)
    semantic_qa_scores = {
        int(idx): 1 / (1 + float(dist))
        for idx, dist in zip(I_qa[0], D_qa[0])
        if idx < len(qa_metadata)
    }

    # ---- QA: BM25 ----
    bm25_qa_raw = qa_bm25.get_scores(query_tokens)
    bm25_qa_top_idx = np.argsort(bm25_qa_raw)[::-1][:top_k * 2]
    max_bm25_qa = bm25_qa_raw[bm25_qa_top_idx[0]] if bm25_qa_raw[bm25_qa_top_idx[0]] > 0 else 1
    bm25_qa_scores = {
        int(i): bm25_qa_raw[i] / max_bm25_qa
        for i in bm25_qa_top_idx
        if bm25_qa_raw[i] > 0
    }

    # ---- QA: Merge ----
    all_qa_idx = set(semantic_qa_scores) | set(bm25_qa_scores)
    merged_qa = {
        i: (semantic_qa_scores.get(i, 0) * 0.6) + (bm25_qa_scores.get(i, 0) * 0.4)
        for i in all_qa_idx
    }
    top_qa = sorted(merged_qa, key=merged_qa.get, reverse=True)[:top_k]
    qa_results = "\n\n".join(qa_metadata[i] for i in top_qa)

    return doc_results, qa_results

# =====================================
# FULL CONTEXT PIPELINE
# =====================================

def retrieve_context(prompt):
    # Step 1: Rewrite query into PQL terminology for better recall
    rewritten = rewrite_query(prompt)
    st.session_state.last_rewritten = rewritten

    # Step 2: Exact function match on original prompt
    exact_match = exact_function_match(prompt)

    # Step 3: Hybrid search using rewritten query
    doc_context, qa_context = hybrid_search(rewritten, top_k=3)

    # Step 4: Assemble full context block
    sections = []

    if exact_match:
        sections.append("### 📖 Official Documentation (Exact Function Match):\n" + exact_match)
    elif doc_context:
        sections.append("### 📖 Official Documentation:\n" + doc_context)

    if qa_context:
        sections.append("### 💡 Relevant PQL Query Examples:\n" + qa_context)

    return "\n\n".join(sections), rewritten

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## 🧠 Celonis Copilot")
    st.markdown("---")
    st.markdown("**Model:** `llama-3.1-8b-instant`")
    st.markdown("**Embed:** `all-MiniLM-L6-v2`")
    st.markdown("**Search:** `Hybrid (Semantic + BM25)`")
    st.markdown("**Query Rewriting:** `✅ Enabled`")
    st.markdown("---")
    st.markdown("**Knowledge Base:**")
    st.markdown(f"- 📚 Docs chunks: `{len(docs_metadata)}`")
    st.markdown(f"- 💡 PQL examples: `{len(qa_metadata)}`")
    st.markdown("---")

    # Show rewritten query for transparency / debugging
    if st.session_state.last_rewritten:
        st.markdown("**🔍 Last Rewritten Query:**")
        st.info(st.session_state.last_rewritten)
        st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_rewritten = ""
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Divyansh")

# =====================================
# UI HEADER
# =====================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Ask anything about **PQL**, **Process Mining**, or **Celonis** platform.")

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build final prompt
    rewritten_query = ""
    if is_celonis_query(prompt):
        context, rewritten_query = retrieve_context(prompt)
        final_prompt = f"""STRICT MODE ENABLED.

Original Question: {prompt}
Rewritten Search Query Used: {rewritten_query}

Documentation & Example Context:
---------------------------------
{context}
---------------------------------

Please answer the original question using the context above.
"""
    else:
        final_prompt = prompt

    # Build clean API message history (original messages for history, injected prompt for current turn)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages[:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": final_prompt})

    # Generate response
    reply = ""
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.1,
                    max_tokens=1500
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
            except Exception as e:
                reply = f"⚠️ Error contacting Groq API: {str(e)}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
