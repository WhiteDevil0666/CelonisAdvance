import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import pickle
import re
import base64
import os
from rank_bm25 import BM25Okapi

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Copilot",
    layout="wide"
)

# =====================================
# BACKGROUND
# =====================================

def set_background(image_file):

    try:

        with open(image_file,"rb") as f:
            encoded=base64.b64encode(f.read()).decode()

        page_bg=f"""
        <style>
        .stApp {{
        background:
        linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
        url("data:image/png;base64,{encoded}");
        background-size:cover;
        background-position:center;
        }}

        h1,h2,h3,p,div,span {{
        color:white !important;
        }}

        .stChatMessage {{
        background:rgba(20,20,30,0.85);
        border-radius:10px;
        padding:12px;
        }}

        </style>
        """

        st.markdown(page_bg,unsafe_allow_html=True)

    except:
        pass

set_background("background.png")

# =====================================
# GROQ CONFIG
# =====================================

MODEL="llama-3.1-8b-instant"

client=Groq(api_key=st.secrets["GROQ_API_KEY"])

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    embed=SentenceTransformer("all-MiniLM-L6-v2")

    reranker=CrossEncoder("BAAI/bge-reranker-base")

    return embed,reranker

EMBED_MODEL,RERANK_MODEL=load_models()

# =====================================
# LOAD VECTOR STORES
# =====================================

@st.cache_resource
def load_kb():

    docs_index=faiss.read_index("pql_faiss.index")

    with open("pql_metadata.pkl","rb") as f:
        docs_metadata=pickle.load(f)

    qa_index=faiss.read_index("pql_knowledge.index")

    with open("pql_knowledge.pkl","rb") as f:
        qa_metadata=pickle.load(f)

    docs_tokens=[re.findall(r"\w+",i["text"].lower()) for i in docs_metadata]

    qa_tokens=[re.findall(r"\w+",i.lower()) for i in qa_metadata]

    docs_bm25=BM25Okapi(docs_tokens)

    qa_bm25=BM25Okapi(qa_tokens)

    return docs_index,docs_metadata,docs_bm25,qa_index,qa_metadata,qa_bm25

docs_index,docs_metadata,docs_bm25,qa_index,qa_metadata,qa_bm25=load_kb()

# =====================================
# SELF LEARNING STORAGE
# =====================================

LEARN_FILE="learned_examples.pkl"

if os.path.exists(LEARN_FILE):

    with open(LEARN_FILE,"rb") as f:
        learned_examples=pickle.load(f)

else:

    learned_examples=[]

# =====================================
# INTENT DETECTION
# =====================================

def detect_example_request(prompt):

    keywords=[
        "example",
        "examples",
        "use case",
        "industry",
        "business",
        "scenario",
        "real world"
    ]

    p=prompt.lower()

    for k in keywords:
        if k in p:
            return True

    return False

# =====================================
# FUNCTION DETECTION
# =====================================

def detect_function(prompt):

    functions=[
    "PU_FIRST","PU_LAST","PU_AVG","PU_SUM","PU_COUNT",
    "DATEDIFF","RUNNING_SUM","COUNT_TABLE"
    ]

    prompt=prompt.upper()

    for f in functions:

        if re.search(rf"\b{f}\b",prompt):
            return f

    return None

# =====================================
# FUNCTION ROUTING
# =====================================

def function_routing(f):

    normalized=f.lower().replace("_","-")

    for item in docs_metadata:

        if normalized in item.get("url","").lower():

            return item["text"]

    return None

# =====================================
# QUERY REWRITE
# =====================================

def rewrite_query(prompt):

    try:

        r=client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role":"user",
                "content":f"Rewrite for Celonis PQL search: {prompt}"
            }],
            temperature=0
        )

        return r.choices[0].message.content.strip()

    except:

        return prompt

# =====================================
# MULTI QUERY
# =====================================

def generate_queries(q):

    try:

        r=client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role":"user",
                "content":f"Generate 3 search queries for: {q}"
            }]
        )

        return r.choices[0].message.content.split("\n")[:3]

    except:

        return []

# =====================================
# HYBRID SEARCH
# =====================================

def hybrid_search(query):

    emb=EMBED_MODEL.encode([query])

    emb=np.array(emb).astype("float32")

    tokens=re.findall(r"\w+",query.lower())

    results=[]

    D,I=docs_index.search(emb,6)

    for s,i in zip(D[0],I[0]):

        if i<len(docs_metadata):

            results.append({
            "text":docs_metadata[i]["text"],
            "score":float(s)*0.7
            })

    bm=docs_bm25.get_scores(tokens)

    top=np.argsort(bm)[::-1][:6]

    for i in top:

        results.append({
        "text":docs_metadata[i]["text"],
        "score":float(bm[i])*0.3
        })

    ranked=sorted(results,key=lambda x:x["score"],reverse=True)

    return [r["text"] for r in ranked[:6]]

# =====================================
# RERANK
# =====================================

def rerank(query,docs):

    pairs=[[query,d] for d in docs]

    scores=RERANK_MODEL.predict(pairs)

    ranked=[d for _,d in sorted(zip(scores,docs),reverse=True)]

    return ranked[:4]

# =====================================
# CONTEXT RETRIEVAL
# =====================================

def retrieve_context(prompt):

    f=detect_function(prompt)

    if f:

        r=function_routing(f)

        if r:
            return r

    rewritten=rewrite_query(prompt)

    queries=[rewritten]+generate_queries(rewritten)

    docs=[]

    for q in queries:

        docs+=hybrid_search(q)

    docs=list(dict.fromkeys(docs))

    docs=rerank(rewritten,docs)

    docs=[d[:1200] for d in docs]

    return "\n\n".join(docs)

# =====================================
# EXPERT MODE
# =====================================

EXPERT_PROMPT="""
You are a senior Celonis Process Mining consultant.

Explain functions using business scenarios.

Provide examples from:

Manufacturing
Finance
Procurement
Supply Chain
Order to Cash
"""

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT="""
You are a Celonis documentation expert.

Answer strictly from context.

If not found say:
Not found in official Celonis documentation.
"""

# =====================================
# SELF LEARNING
# =====================================

def store_learning(question,answer):

    if len(answer)>200:

        learned_examples.append({
        "q":question,
        "a":answer
        })

        with open(LEARN_FILE,"wb") as f:
            pickle.dump(learned_examples,f)

# =====================================
# CHAT STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages=[]

# =====================================
# UI
# =====================================

st.title("🧠 Celonis Copilot")

# =====================================
# HISTORY
# =====================================

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt:=st.chat_input("Ask Celonis question..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):

        st.markdown(prompt)

    example_mode=detect_example_request(prompt)

    if example_mode:

        messages=[
        {"role":"system","content":EXPERT_PROMPT},
        {"role":"user","content":prompt}
        ]

    else:

        context=retrieve_context(prompt)

        final_prompt=f"""
Question:
{prompt}

Context:
{context}
"""

        messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":final_prompt}
        ]

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            r=client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
            )

            answer=r.choices[0].message.content

            st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})

    store_learning(prompt,answer)
