import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle
import re
import os
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="Celonis AI Agent", layout="wide")

MODEL_FAST = "llama-3.1-8b-instant"
MODEL_REASON = "llama-3.1-70b-versatile"

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    rerank = CrossEncoder("BAAI/bge-reranker-base")
    return embed, rerank

EMBED_MODEL, RERANK_MODEL = load_models()

# ==========================================================
# LOAD KNOWLEDGE BASE
# ==========================================================

def load_index(index_file, meta_file):

    if os.path.exists(index_file):

        index = faiss.read_index(index_file)

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

    else:

        index = faiss.IndexFlatL2(384)
        meta = []

    return index, meta

docs_index, docs_meta = load_index("pql_faiss.index","pql_metadata.pkl")
qa_index, qa_meta = load_index("pql_knowledge.index","pql_knowledge.pkl")

dynamic_index, dynamic_meta = load_index("dynamic_docs.index","dynamic_docs.pkl")

# ==========================================================
# BM25
# ==========================================================

docs_tokens = [re.findall(r"\w+", i["text"].lower()) for i in docs_meta] if docs_meta else []
docs_bm25 = BM25Okapi(docs_tokens) if docs_tokens else None

# ==========================================================
# SELF LEARNING MEMORY
# ==========================================================

LEARN_FILE = "learned_examples.pkl"

if os.path.exists(LEARN_FILE):

    with open(LEARN_FILE,"rb") as f:
        learned = pickle.load(f)

else:

    learned = []

def store_learning(q,a):

    if len(a) < 200:
        return

    learned.append({"q":q,"a":a})

    with open(LEARN_FILE,"wb") as f:
        pickle.dump(learned,f)

# ==========================================================
# DOC SCRAPER
# ==========================================================

def scrape_celonis(query):

    try:

        search_url=f"https://docs.celonis.com/en/search.html?q={query}"

        r=requests.get(search_url,timeout=10)

        soup=BeautifulSoup(r.text,"html.parser")

        links=[]

        for a in soup.find_all("a",href=True):

            if "/en/" in a["href"] and ".html" in a["href"]:

                links.append("https://docs.celonis.com"+a["href"])

        if not links:
            return None

        page=requests.get(links[0],timeout=10)

        ps=BeautifulSoup(page.text,"html.parser").find_all("p")

        text="\n".join(p.get_text() for p in ps)

        return text[:6000]

    except:

        return None

# ==========================================================
# VECTOR SEARCH
# ==========================================================

def vector_search(query, index, meta, top_k=4):

    if len(meta)==0:
        return []

    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb).astype("float32")

    D,I = index.search(emb,top_k)

    docs=[]

    for idx in I[0]:

        if idx < len(meta):

            item=meta[idx]

            if isinstance(item,dict):
                docs.append(item["text"])
            else:
                docs.append(item)

    return docs

# ==========================================================
# RERANK
# ==========================================================

def rerank(query, docs):

    if not docs:
        return []

    pairs=[[query,d] for d in docs]

    scores=RERANK_MODEL.predict(pairs)

    ranked=[d for _,d in sorted(zip(scores,docs),reverse=True)]

    return ranked[:4]

# ==========================================================
# RETRIEVAL AGENT
# ==========================================================

def retrieve_docs(query):

    docs=[]

    docs += vector_search(query,docs_index,docs_meta)
    docs += vector_search(query,qa_index,qa_meta)
    docs += vector_search(query,dynamic_index,dynamic_meta)

    docs=list(dict.fromkeys(docs))

    docs=rerank(query,docs)

    if not docs:

        scraped=scrape_celonis(query)

        if scraped:

            chunks=[scraped[i:i+500] for i in range(0,len(scraped),500)]

            emb=EMBED_MODEL.encode(chunks)
            emb=np.array(emb).astype("float32")

            dynamic_index.add(emb)

            dynamic_meta.extend(chunks)

            faiss.write_index(dynamic_index,"dynamic_docs.index")

            with open("dynamic_docs.pkl","wb") as f:
                pickle.dump(dynamic_meta,f)

            docs=chunks[:4]

    return "\n\n".join(docs)

# ==========================================================
# INTENT AGENT
# ==========================================================

def classify_intent(prompt):

    p=prompt.lower()

    if "write" in p or "generate" in p or "query" in p:
        return "pql_generation"

    if "what is" in p or "explain" in p:
        return "explanation"

    if "process" in p or "cycle time" in p or "throughput":
        return "business_problem"

    return "general"

# ==========================================================
# PQL GENERATOR AGENT
# ==========================================================

def generate_pql(prompt, context):

    system="""
You are a Celonis Process Mining expert.

Write correct PQL.

Rules:
- Never use SQL
- Quote tables and columns "Table"."Column"
- Use correct Celonis functions
"""

    user=f"""
Question:
{prompt}

Reference:
{context}

Write the PQL query and explanation.
"""

    r=client.chat.completions.create(
        model=MODEL_REASON,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.2,
        max_tokens=1200
    )

    return r.choices[0].message.content

# ==========================================================
# EXPLANATION AGENT
# ==========================================================

def explain(prompt, context):

    r=client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role":"system","content":"Explain Celonis concepts clearly."},
            {"role":"user","content":f"{prompt}\n\n{context}"}
        ]
    )

    return r.choices[0].message.content

# ==========================================================
# VALIDATOR
# ==========================================================

def validate_pql(answer):

    sql_words=["SELECT","FROM","WHERE","JOIN"]

    for w in sql_words:

        if w in answer.upper():

            return False

    return True

# ==========================================================
# QUERY PLANNER AGENT
# ==========================================================

def planner(prompt):

    intent=classify_intent(prompt)

    if intent=="pql_generation":
        return ["retrieve_docs","generate_pql"]

    if intent=="explanation":
        return ["retrieve_docs","explain"]

    if intent=="business_problem":
        return ["retrieve_docs","generate_pql"]

    return ["explain"]

# ==========================================================
# EXECUTOR
# ==========================================================

def run_agent(prompt):

    plan=planner(prompt)

    context=""

    if "retrieve_docs" in plan:

        context=retrieve_docs(prompt)

    if "generate_pql" in plan:

        ans=generate_pql(prompt,context)

    else:

        ans=explain(prompt,context)

    if not validate_pql(ans):

        ans="⚠ Generated SQL instead of PQL. Please refine the query."

    return ans

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🧠 Celonis Process Mining AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt=st.chat_input("Ask about Celonis, PQL, or your business process")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Agent thinking..."):

            answer=run_agent(prompt)

            st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})

    store_learning(prompt,answer)
