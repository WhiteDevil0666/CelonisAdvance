import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
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
# BACKGROUND
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        page_bg = f"""
        <style>

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

    except Exception:
        pass  # FIX: bare except replaced with except Exception

set_background("background.png")

# =====================================
# CONFIG
# =====================================

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    # NOTE: First load downloads ~1GB — will be slow once, then cached
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embed, reranker

EMBED_MODEL, RERANK_MODEL = load_models()

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant.

Rules:
- Use provided documentation context strictly.
- Preserve official PQL syntax EXACTLY.
- Do NOT convert PQL into SQL.
- Do NOT invent functions or examples.
- Never use SQL keywords like SELECT, FROM, WHERE, JOIN.
- If answer not found in documentation say:
  "Not found in official Celonis documentation."
- Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# LOAD VECTOR STORES + BM25
# =====================================

@st.cache_resource
def load_all_stores():

    # --- Official Docs ---
    docs_index = faiss.read_index("pql_faiss.index")
    with open("pql_metadata.pkl", "rb") as f:
        docs_metadata = pickle.load(f)     # list of dicts: {url, title, text}

    # --- PQL Q&A Examples ---
    qa_index = faiss.read_index("pql_knowledge.index")
    with open("pql_knowledge.pkl", "rb") as f:
        qa_metadata = pickle.load(f)       # list of strings

    # BM25 uses proper tokenization with re.findall (better than .split())
    docs_corpus = [re.findall(r"\w+", item["text"].lower()) for item in docs_metadata]
    qa_corpus   = [re.findall(r"\w+", item.lower()) for item in qa_metadata]

    docs_bm25 = BM25Okapi(docs_corpus)
    qa_bm25   = BM25Okapi(qa_corpus)

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
# FUNCTION DETECTION
# =====================================

def detect_pql_function(prompt):
    """Detect if a known PQL function is explicitly mentioned in the prompt."""
    functions = [
        "PU_FIRST", "PU_LAST", "PU_AVG", "PU_SUM", "PU_COUNT",
        "PU_MIN", "PU_MAX", "PU_COUNT_DISTINCT",
        "DATEDIFF", "RUNNING_SUM", "COUNT_TABLE",
        "MOVING_AVG", "REMAP", "AVG", "MEDIAN",
        "SOURCE", "TARGET", "RUNNING_TOTAL"
    ]
    prompt_upper = prompt.upper()
    for f in functions:
        if f in prompt_upper:
            return f
    return None

# =====================================
# FUNCTION ROUTING
# =====================================

def function_routing(function_name):
    """
    Directly route to the doc chunk whose URL matches the function name.
    FIX: Normalizes _ to - so RUNNING_SUM matches running-sum.html correctly.
    """
    normalized = function_name.lower().replace("_", "-")

    for item in docs_metadata:
        url = item.get("url", "").lower()
        if normalized in url or function_name.lower() in url:
            return item["text"]

    return None

# =====================================
# QUERY REWRITING
# =====================================

def rewrite_query(prompt):
    """Rewrite user question into precise PQL/Celonis terminology for better retrieval."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""You are a Celonis PQL expert.
Rewrite this question into precise PQL and process mining terminology
for searching a Celonis documentation database.

Rules:
- Use official PQL function names (e.g., PU_AVG, DATEDIFF, COUNT_TABLE, RUNNING_SUM)
- Use Celonis terms (e.g., case table, activity table, event log, pull-up functions)
- Return ONLY the rewritten query, nothing else
- Keep it concise (1-2 sentences)

Question: {prompt}

Rewritten Query:"""
            }],
            temperature=0,
            max_tokens=150   # FIX: added max_tokens to prevent quota waste
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return prompt  # FIX: bare except replaced with except Exception

# =====================================
# MULTI-QUERY GENERATION
# =====================================

def generate_multi_queries(query):
    """Generate 3 alternative phrasings to improve retrieval recall."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""Generate 3 alternative search queries for Celonis documentation.
Each should approach the topic differently to improve search recall.

Original Query: {query}

Return exactly 3 queries, one per line, no numbering or bullets."""
            }],
            temperature=0.3,
            max_tokens=150   # FIX: added max_tokens
        )
        queries = response.choices[0].message.content.strip().split("\n")
        return [q.strip() for q in queries if q.strip()][:3]

    except Exception:
        return []  # FIX: bare except replaced

# =====================================
# HYBRID SEARCH (SEMANTIC + BM25)
# =====================================

def hybrid_search(query, top_k=5):
    """
    FIX: Now searches BOTH docs store AND qa store (previously only searched docs).
    Combines FAISS semantic search + BM25 keyword search on both stores.
    """
    query_embedding = EMBED_MODEL.encode([query])
    query_np = np.array(query_embedding).astype("float32")
    tokens = re.findall(r"\w+", query.lower())

    results = []

    # ---- DOCS: Semantic ----
    D_docs, I_docs = docs_index.search(query_np, top_k)
    for idx in I_docs[0]:
        if idx < len(docs_metadata):
            results.append(docs_metadata[idx]["text"])

    # ---- DOCS: BM25 ----
    bm25_doc_scores = docs_bm25.get_scores(tokens)
    top_doc_bm25 = np.argsort(bm25_doc_scores)[::-1][:top_k]
    for i in top_doc_bm25:
        if bm25_doc_scores[i] > 0:
            results.append(docs_metadata[i]["text"])

    # ---- QA: Semantic ---- FIX: was completely missing before
    D_qa, I_qa = qa_index.search(query_np, top_k)
    for idx in I_qa[0]:
        if idx < len(qa_metadata):
            results.append(qa_metadata[idx])

    # ---- QA: BM25 ---- FIX: was completely missing before
    bm25_qa_scores = qa_bm25.get_scores(tokens)
    top_qa_bm25 = np.argsort(bm25_qa_scores)[::-1][:top_k]
    for i in top_qa_bm25:
        if bm25_qa_scores[i] > 0:
            results.append(qa_metadata[i])

    return results

# =====================================
# DEDUPLICATION (ORDER-PRESERVING)
# =====================================

def deduplicate(docs):
    """
    FIX: Replaced set() which destroyed order.
    Uses first-100-char fingerprint to deduplicate while preserving order.
    """
    seen = set()
    unique = []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

# =====================================
# RERANKING
# =====================================

def rerank_results(query, docs, top_k=4):
    """
    FIX: Added empty list guard to prevent crash when docs is empty.
    Uses CrossEncoder to score and rank all retrieved candidates.
    """
    if not docs:
        return []

    pairs = [[query, doc] for doc in docs]
    scores = RERANK_MODEL.predict(pairs)
    ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return ranked[:top_k]

# =====================================
# FULL CONTEXT PIPELINE
# =====================================

def retrieve_context(prompt):
    """
    Pipeline:
    1. Exact function routing (fastest, most precise)
    2. Query rewriting into PQL terminology
    3. Multi-query generation for better recall
    4. Hybrid search (semantic + BM25) on BOTH stores
    5. Deduplication (order-preserving)
    6. Cross-encoder reranking for best results
    """

    # Step 1: Direct function routing
    func = detect_pql_function(prompt)
    if func:
        routed_doc = function_routing(func)
        if routed_doc:
            # FIX: Update sidebar state even on routing path
            st.session_state.last_rewritten = f"[Direct routing → {func}]"
            return routed_doc, func

    # Step 2: Rewrite query
    rewritten = rewrite_query(prompt)
    st.session_state.last_rewritten = rewritten

    # Step 3: Generate alternative queries
    alt_queries = generate_multi_queries(rewritten)
    all_queries = [rewritten] + alt_queries

    # Step 4: Hybrid search across all queries on both stores
    all_docs = []
    for q in all_queries:
        all_docs += hybrid_search(q, top_k=4)

    # Step 5: Deduplicate (order-preserving)
    all_docs = deduplicate(all_docs)

    # Step 6: Rerank with CrossEncoder
    top_docs = rerank_results(rewritten, all_docs, top_k=4)

    context = "\n\n".join(top_docs)

    return context[:8000], rewritten

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## 🧠 Celonis Copilot")
    st.markdown("---")
    st.markdown("**Pipeline:**")
    st.markdown("- 🔀 Function-Aware Routing")
    st.markdown("- ✍️ LLM Query Rewriting")
    st.markdown("- 🔁 Multi-Query Generation")
    st.markdown("- 🔍 Hybrid Search (Semantic + BM25)")
    st.markdown("- 🏆 Cross-Encoder Reranking")
    st.markdown("---")
    st.markdown("**Knowledge Base:**")
    st.markdown(f"- 📚 Docs chunks: `{len(docs_metadata)}`")
    st.markdown(f"- 💡 PQL examples: `{len(qa_metadata)}`")
    st.markdown("---")

    if st.session_state.last_rewritten:
        st.markdown("**🔍 Last Rewritten Query:**")
        st.info(st.session_state.last_rewritten)
        st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_rewritten = ""
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Divyansh")

# =====================================
# UI HEADER
# =====================================

st.title("🧠 Celonis Process Mining Copilot")
st.markdown("Ask questions about **PQL**, **Process Mining**, or **Celonis**")

# =====================================
# CHAT HISTORY
# =====================================

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    context, rewritten = retrieve_context(prompt)

    final_prompt = f"""Original Question:
{prompt}

Rewritten Search Query Used:
{rewritten}

Documentation & Example Context:
---------------------------------
{context}
---------------------------------

Answer the original question using ONLY the provided documentation context above.
"""

    # Build clean API message history
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[:-1]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": final_prompt})

    # Generate response
    reply = ""
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:  # FIX: Added error handling — previously could crash silently
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
