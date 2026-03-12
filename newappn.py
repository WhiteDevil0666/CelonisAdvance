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
            background-color: rgba(20,20,30,0.85);
            border-radius: 12px;
            padding: 12px;
        }}

        div[data-testid="stChatInput"] {{
            background-color: rgba(20,20,30,0.95);
            border-radius: 12px;
            padding: 8px;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(15,15,25,0.95);
        }}

        h1,h2,h3,p,div,span {{
            color:white !important;
        }}

        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)

    except Exception:
        pass

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
- If answer not found say:
  "Not found in official Celonis documentation."
- Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# LOAD VECTOR STORES
# =====================================

@st.cache_resource
def load_all_stores():

    docs_index = faiss.read_index("pql_faiss.index")

    with open("pql_metadata.pkl","rb") as f:
        docs_metadata = pickle.load(f)

    qa_index = faiss.read_index("pql_knowledge.index")

    with open("pql_knowledge.pkl","rb") as f:
        qa_metadata = pickle.load(f)

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
# FUNCTION DETECTION (REGEX)
# =====================================

def detect_pql_function(prompt):

    functions = [
        "PU_FIRST","PU_LAST","PU_AVG","PU_SUM","PU_COUNT",
        "PU_MIN","PU_MAX","PU_COUNT_DISTINCT",
        "DATEDIFF","RUNNING_SUM","COUNT_TABLE",
        "MOVING_AVG","REMAP","AVG","MEDIAN",
        "SOURCE","TARGET","RUNNING_TOTAL"
    ]

    prompt_upper = prompt.upper()

    for f in functions:
        pattern = rf"\b{f}\b"
        if re.search(pattern, prompt_upper):
            return f

    return None

# =====================================
# FUNCTION ROUTING
# =====================================

def function_routing(function_name):

    normalized = function_name.lower().replace("_","-")

    for item in docs_metadata:
        url = item.get("url","").lower()

        if normalized in url or function_name.lower() in url:
            return item["text"]

    return None

# =====================================
# QUERY REWRITE
# =====================================

def rewrite_query(prompt):

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role":"user",
                "content":f"""
Rewrite the question into Celonis PQL terminology.

Question: {prompt}

Return only rewritten query.
"""
            }],
            temperature=0,
            max_tokens=120
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return prompt

# =====================================
# MULTI QUERY GENERATION
# =====================================

def generate_multi_queries(query):

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role":"user",
                "content":f"""
Generate 3 alternative search queries.

Original Query:
{query}

Return 3 lines only.
"""
            }],
            temperature=0.3,
            max_tokens=120
        )

        queries = response.choices[0].message.content.strip().split("\n")

        return [q.strip() for q in queries if q.strip()][:3]

    except Exception:
        return []

# =====================================
# WEIGHTED HYBRID SEARCH
# =====================================

def hybrid_search(query, top_k=6):

    query_embedding = EMBED_MODEL.encode([query])
    query_np = np.array(query_embedding).astype("float32")

    tokens = re.findall(r"\w+", query.lower())

    results = []

    # ===== DOCS SEMANTIC =====
    D_docs, I_docs = docs_index.search(query_np, top_k)

    for score, idx in zip(D_docs[0], I_docs[0]):

        if idx < len(docs_metadata):

            results.append({
                "text": docs_metadata[idx]["text"],
                "score": float(score) * 0.7
            })

    # ===== DOCS BM25 =====
    bm25_scores = docs_bm25.get_scores(tokens)

    top_doc_bm25 = np.argsort(bm25_scores)[::-1][:top_k]

    for i in top_doc_bm25:

        if bm25_scores[i] > 0:

            results.append({
                "text": docs_metadata[i]["text"],
                "score": float(bm25_scores[i]) * 0.3
            })

    # ===== QA SEMANTIC =====
    D_qa, I_qa = qa_index.search(query_np, top_k)

    for score, idx in zip(D_qa[0], I_qa[0]):

        if idx < len(qa_metadata):

            results.append({
                "text": qa_metadata[idx],
                "score": float(score) * 0.7
            })

    # ===== QA BM25 =====
    bm25_scores = qa_bm25.get_scores(tokens)

    top_qa_bm25 = np.argsort(bm25_scores)[::-1][:top_k]

    for i in top_qa_bm25:

        if bm25_scores[i] > 0:

            results.append({
                "text": qa_metadata[i],
                "score": float(bm25_scores[i]) * 0.3
            })

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    return [r["text"] for r in ranked[:top_k]]

# =====================================
# DEDUPLICATION
# =====================================

def deduplicate(docs):

    seen=set()
    unique=[]

    for doc in docs:

        key=doc[:120]

        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique

# =====================================
# RERANK
# =====================================

def rerank_results(query,docs,top_k=4):

    if not docs:
        return []

    pairs=[[query,doc] for doc in docs]

    scores=RERANK_MODEL.predict(pairs)

    ranked=[doc for _,doc in sorted(zip(scores,docs),reverse=True)]

    return ranked[:top_k]

# =====================================
# RETRIEVAL PIPELINE
# =====================================

def retrieve_context(prompt):

    func = detect_pql_function(prompt)

    if func:
        routed_doc=function_routing(func)

        if routed_doc:
            st.session_state.last_rewritten=f"[Function Routing → {func}]"
            return routed_doc,func

    rewritten = rewrite_query(prompt)

    st.session_state.last_rewritten = rewritten

    alt_queries = generate_multi_queries(rewritten)

    all_queries = [rewritten] + alt_queries

    all_docs=[]

    for q in all_queries:
        all_docs += hybrid_search(q)

    all_docs = deduplicate(all_docs)

    top_docs = rerank_results(rewritten, all_docs)

    compressed_docs=[doc[:1200] for doc in top_docs]

    context="\n\n".join(compressed_docs)

    return context, rewritten

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.markdown("## 🧠 Celonis Copilot")

    st.markdown("---")

    st.markdown("Pipeline:")

    st.markdown("""
Function Routing  
Query Rewrite  
Multi Query Generation  
Hybrid Retrieval  
Cross Encoder Rerank  
Context Compression
""")

    st.markdown("---")

    st.markdown(f"Docs chunks: {len(docs_metadata)}")

    st.markdown(f"PQL examples: {len(qa_metadata)}")

    if st.session_state.last_rewritten:

        st.markdown("Rewritten Query:")

        st.info(st.session_state.last_rewritten)

    if st.button("Clear Chat"):

        st.session_state.messages=[]

        st.session_state.last_rewritten=""

        st.rerun()

    st.caption("Powered by Divyansh")

# =====================================
# UI
# =====================================

st.title("🧠 Celonis Process Mining Copilot")

st.markdown("Ask questions about **PQL**, **Celonis**, or **Process Mining**")

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

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context, rewritten = retrieve_context(prompt)

    final_prompt=f"""
Original Question:
{prompt}

Search Query Used:
{rewritten}

Documentation Context:
{context}

Answer using ONLY the documentation above.
"""

    api_messages=[{"role":"system","content":SYSTEM_PROMPT}]

    for m in st.session_state.messages[:-1]:

        api_messages.append({
            "role":m["role"],
            "content":m["content"]
        })

    api_messages.append({
        "role":"user",
        "content":final_prompt
    })

    reply=""

    with st.chat_message("assistant"):

        with st.spinner("Analyzing..."):

            try:

                response=client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.1,
                    max_tokens=1500
                )

                reply=response.choices[0].message.content

                st.markdown(reply)

            except Exception as e:

                reply=f"Error contacting Groq API: {str(e)}"

                st.error(reply)

    st.session_state.messages.append({
        "role":"assistant",
        "content":reply
    })
