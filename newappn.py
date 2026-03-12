import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import pickle
import re
import os
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Celonis Copilot", layout="wide")

# =========================================================
# CONFIG
# =========================================================

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

STATIC_INDEX = "pql_faiss.index"
STATIC_META = "pql_metadata.pkl"

QA_INDEX = "pql_knowledge.index"
QA_META = "pql_knowledge.pkl"

DYNAMIC_INDEX = "dynamic_docs.index"
DYNAMIC_META = "dynamic_docs.pkl"

LEARN_FILE = "learned_examples.pkl"

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    rerank = CrossEncoder("BAAI/bge-reranker-base")
    return embed, rerank

EMBED_MODEL, RERANK_MODEL = load_models()

# =========================================================
# LOAD STATIC KNOWLEDGE BASE
# =========================================================

@st.cache_resource
def load_static_kb():

    docs_index = faiss.read_index(STATIC_INDEX)

    with open(STATIC_META,"rb") as f:
        docs_meta = pickle.load(f)

    qa_index = faiss.read_index(QA_INDEX)

    with open(QA_META,"rb") as f:
        qa_meta = pickle.load(f)

    docs_tokens = [re.findall(r"\w+", i["text"].lower()) for i in docs_meta]
    qa_tokens   = [re.findall(r"\w+", i.lower()) for i in qa_meta]

    docs_bm25 = BM25Okapi(docs_tokens)
    qa_bm25   = BM25Okapi(qa_tokens)

    return docs_index, docs_meta, docs_bm25, qa_index, qa_meta, qa_bm25

docs_index, docs_meta, docs_bm25, qa_index, qa_meta, qa_bm25 = load_static_kb()

# =========================================================
# LOAD DYNAMIC STORE
# =========================================================

def load_dynamic():

    if os.path.exists(DYNAMIC_INDEX):

        index = faiss.read_index(DYNAMIC_INDEX)

        with open(DYNAMIC_META,"rb") as f:
            meta = pickle.load(f)

    else:

        index = faiss.IndexFlatL2(384)
        meta = []

    return index, meta

dynamic_index, dynamic_meta = load_dynamic()

# =========================================================
# LOAD LEARNING MEMORY
# =========================================================

if os.path.exists(LEARN_FILE):
    with open(LEARN_FILE,"rb") as f:
        learned_examples = pickle.load(f)
else:
    learned_examples = []

# =========================================================
# UTIL FUNCTIONS
# =========================================================

def chunk_text(text, size=500):

    chunks = []

    for i in range(0,len(text),size):
        chunks.append(text[i:i+size])

    return chunks


def add_dynamic_docs(text):

    chunks = chunk_text(text)

    emb = EMBED_MODEL.encode(chunks)
    emb = np.array(emb).astype("float32")

    dynamic_index.add(emb)

    dynamic_meta.extend(chunks)

    faiss.write_index(dynamic_index, DYNAMIC_INDEX)

    with open(DYNAMIC_META,"wb") as f:
        pickle.dump(dynamic_meta,f)


def search_dynamic(query, top_k=3):

    if len(dynamic_meta) == 0:
        return []

    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb).astype("float32")

    D,I = dynamic_index.search(emb, top_k)

    docs = []

    for idx in I[0]:
        if idx < len(dynamic_meta):
            docs.append(dynamic_meta[idx])

    return docs


def store_learning(q,a):

    if len(a) < 200:
        return

    learned_examples.append({"q":q,"a":a})

    with open(LEARN_FILE,"wb") as f:
        pickle.dump(learned_examples,f)


# =========================================================
# DOC SCRAPER
# =========================================================

def fetch_celonis_doc(query):

    try:

        search = f"https://docs.celonis.com/en/search.html?q={query.replace(' ','%20')}"

        r = requests.get(search,timeout=10)

        soup = BeautifulSoup(r.text,"html.parser")

        links = []

        for a in soup.find_all("a",href=True):

            if ".html" in a["href"] and "/en/" in a["href"]:
                links.append("https://docs.celonis.com"+a["href"])

        if not links:
            return None

        page = requests.get(links[0],timeout=10)

        ps = BeautifulSoup(page.text,"html.parser").find_all("p")

        text = "\n".join(p.get_text() for p in ps)

        return text[:6000]

    except:
        return None

# =========================================================
# QUERY REWRITE
# =========================================================

def rewrite_query(q):

    try:

        r = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role":"user",
                "content":f"Rewrite for Celonis documentation search: {q}"
            }],
            temperature=0
        )

        return r.choices[0].message.content.strip()

    except:

        return q

# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(query):

    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb).astype("float32")

    tokens = re.findall(r"\w+",query.lower())

    results = []

    # semantic docs
    D,I = docs_index.search(emb,5)

    for dist,idx in zip(D[0],I[0]):

        if idx < len(docs_meta):

            score = 1/(1+dist)

            results.append((score,docs_meta[idx]["text"]))

    # bm25 docs

    bm = docs_bm25.get_scores(tokens)

    top = np.argsort(bm)[::-1][:5]

    for i in top:

        results.append((bm[i],docs_meta[i]["text"]))

    # semantic QA

    D2,I2 = qa_index.search(emb,3)

    for dist,idx in zip(D2[0],I2[0]):

        if idx < len(qa_meta):

            results.append((1/(1+dist), qa_meta[idx]))

    ranked = sorted(results,key=lambda x:x[0],reverse=True)

    return [r[1] for r in ranked[:6]]

# =========================================================
# RERANK
# =========================================================

def rerank(query, docs):

    if not docs:
        return []

    pairs = [[query,d] for d in docs]

    scores = RERANK_MODEL.predict(pairs)

    ranked = [d for _,d in sorted(zip(scores,docs),reverse=True)]

    return ranked[:4]

# =========================================================
# RETRIEVAL PIPELINE
# =========================================================

def retrieve_context(prompt):

    rewritten = rewrite_query(prompt)

    docs = hybrid_search(rewritten)

    docs = rerank(rewritten,docs)

    dynamic = search_dynamic(prompt)

    docs.extend(dynamic)

    docs = list(dict.fromkeys(docs))

    if not docs:

        scraped = fetch_celonis_doc(prompt)

        if scraped:

            add_dynamic_docs(scraped)

            docs = search_dynamic(prompt)

    return "\n\n".join(d[:1200] for d in docs)

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a Celonis Process Mining expert and PQL specialist.

Rules:

- Use PQL syntax correctly
- Never use SQL syntax
- Quote tables and columns:
  "Table"."Column"

Explain clearly and concisely.

Always verify the query before explaining it.
"""

# =========================================================
# CHAT STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages=[]

# =========================================================
# UI
# =========================================================

st.title("🧠 Celonis Process Mining Copilot")

st.caption("Hybrid RAG + Self-Learning + Auto Documentation")

# =========================================================
# CHAT HISTORY
# =========================================================

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# =========================================================
# INPUT
# =========================================================

if prompt := st.chat_input("Ask about Celonis, PQL or Process Mining"):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context = retrieve_context(prompt)

    final_prompt = f"""
Question:
{prompt}

Reference:
{context}

Answer using the reference when available.
"""

    messages = [{"role":"system","content":SYSTEM_PROMPT}]

    for m in st.session_state.messages[:-1]:

        messages.append({
            "role":m["role"],
            "content":m["content"]
        })

    messages.append({"role":"user","content":final_prompt})

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            r = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )

            answer = r.choices[0].message.content

            st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})

    store_learning(prompt,answer)
