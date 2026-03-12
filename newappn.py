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
        }}

        .stChatMessage {{
            background-color: rgba(20,20,30,0.85);
            border-radius: 12px;
            padding: 12px;
        }}

        h1,h2,h3,p,div,span {{
            color:white !important;
        }}

        </style>
        """

        st.markdown(page_bg, unsafe_allow_html=True)

    except:
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

- Use provided documentation context strictly
- Preserve official PQL syntax
- Do NOT convert PQL into SQL
- Do NOT invent functions
- If not found in documentation say:
  "Not found in official Celonis documentation."
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

    docs_corpus = [
        re.findall(r"\w+", item["text"].lower())
        for item in docs_metadata
    ]

    qa_corpus = [
        re.findall(r"\w+", item.lower())
        for item in qa_metadata
    ]

    docs_bm25 = BM25Okapi(docs_corpus)

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
# FUNCTION DETECTION
# =====================================

def detect_pql_function(prompt):

    functions = [
        "PU_FIRST","PU_LAST","PU_AVG","PU_SUM",
        "DATEDIFF","RUNNING_SUM","COUNT_TABLE",
        "PU_COUNT","PU_MIN","PU_MAX"
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

    for item in docs_metadata:

        url = item.get("url","").upper()

        if function_name in url:

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
Rewrite the question for Celonis documentation search.

Use official PQL terminology.

Question: {prompt}

Return only rewritten query.
"""
            }],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except:

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
Generate 3 alternative search queries for Celonis documentation.

Query:
{query}

Return each query on new line.
"""
            }],
            temperature=0
        )

        queries = response.choices[0].message.content.split("\n")

        return [q.strip() for q in queries if q.strip()]

    except:

        return []

# =====================================
# HYBRID SEARCH
# =====================================

def hybrid_search(query):

    query_embedding = EMBED_MODEL.encode([query])

    query_np = np.array(query_embedding).astype("float32")

    tokens = re.findall(r"\w+", query.lower())

    D,I = docs_index.search(query_np,5)

    docs = []

    for idx in I[0]:

        if idx < len(docs_metadata):

            docs.append(docs_metadata[idx]["text"])

    bm25_scores = docs_bm25.get_scores(tokens)

    top = np.argsort(bm25_scores)[::-1][:5]

    for i in top:

        docs.append(docs_metadata[i]["text"])

    return docs

# =====================================
# RERANKING
# =====================================

def rerank_results(query, docs):

    pairs = [[query,doc] for doc in docs]

    scores = RERANK_MODEL.predict(pairs)

    ranked = [
        doc for _,doc in sorted(zip(scores,docs),reverse=True)
    ]

    return ranked[:3]

# =====================================
# CONTEXT RETRIEVAL
# =====================================

def retrieve_context(prompt):

    # FUNCTION ROUTING
    func = detect_pql_function(prompt)

    if func:

        routed_doc = function_routing(func)

        if routed_doc:

            return routed_doc, func

    # QUERY REWRITE
    rewritten = rewrite_query(prompt)

    st.session_state.last_rewritten = rewritten

    queries = [rewritten] + generate_multi_queries(rewritten)

    docs = []

    for q in queries:

        docs += hybrid_search(q)

    docs = list(set(docs))

    docs = rerank_results(rewritten, docs)

    context = "\n\n".join(docs)

    return context[:8000], rewritten

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.markdown("## 🧠 Celonis Copilot")

    st.markdown("---")

    st.markdown("Hybrid Retrieval")

    st.markdown("Multi Query Retrieval")

    st.markdown("Cross Encoder Reranking")

    st.markdown("Function Aware Routing")

    st.markdown("---")

    if st.session_state.last_rewritten:

        st.markdown("**Last Rewritten Query**")

        st.info(st.session_state.last_rewritten)

    if st.button("Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

# =====================================
# UI
# =====================================

st.title("Celonis Process Mining Copilot")

st.markdown("Ask questions about **PQL**, **Process Mining**, or **Celonis**")

# =====================================
# CHAT HISTORY
# =====================================

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# =====================================
# USER INPUT
# =====================================

if prompt := st.chat_input("Ask a question"):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):

        st.markdown(prompt)

    context = ""

    context, rewritten = retrieve_context(prompt)

    final_prompt = f"""
Original Question:
{prompt}

Documentation Context:
{context}

Answer using only the provided documentation.
"""

    api_messages = [{"role":"system","content":SYSTEM_PROMPT}]

    for m in st.session_state.messages[:-1]:

        api_messages.append(m)

    api_messages.append({"role":"user","content":final_prompt})

    with st.chat_message("assistant"):

        with st.spinner("Analyzing..."):

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.1,
                max_tokens=1500
            )

            reply = response.choices[0].message.content

            st.markdown(reply)

    st.session_state.messages.append({"role":"assistant","content":reply})
