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
    page_title="Celonis PQL Copilot",
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
                linear-gradient(rgba(0,0,0,0.88), rgba(0,0,0,0.88)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{ background-color: transparent !important; }}
        section.main > div {{ background-color: transparent !important; }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}

        .stChatMessage {{
            background: rgba(20, 20, 35, 0.88);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 8px;
        }}

        div[data-testid="stChatInput"] {{
            background-color: rgba(20, 20, 35, 0.95);
            border-radius: 12px;
            padding: 8px;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(12, 12, 22, 0.97);
        }}

        h1, h2, h3, p, div, span, label {{
            color: white !important;
        }}

        code {{
            background: rgba(255,255,255,0.1) !important;
            color: #a8d8ff !important;
            border-radius: 4px;
            padding: 2px 6px;
        }}

        pre {{
            background: rgba(0,0,0,0.5) !important;
            border-radius: 8px;
            padding: 12px !important;
        }}

        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)

    except Exception:
        pass

set_background("background.png")

# =====================================
# GROQ CONFIG
# =====================================

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():
    embed    = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embed, reranker

EMBED_MODEL, RERANK_MODEL = load_models()

# =====================================
# LOAD KNOWLEDGE BASE
# =====================================

@st.cache_resource
def load_kb():
    docs_index = faiss.read_index("pql_faiss.index")
    with open("pql_metadata.pkl", "rb") as f:
        docs_metadata = pickle.load(f)

    qa_index = faiss.read_index("pql_knowledge.index")
    with open("pql_knowledge.pkl", "rb") as f:
        qa_metadata = pickle.load(f)

    docs_tokens = [re.findall(r"\w+", i["text"].lower()) for i in docs_metadata]
    qa_tokens   = [re.findall(r"\w+", i.lower()) for i in qa_metadata]

    docs_bm25 = BM25Okapi(docs_tokens)
    qa_bm25   = BM25Okapi(qa_tokens)

    return docs_index, docs_metadata, docs_bm25, qa_index, qa_metadata, qa_bm25

docs_index, docs_metadata, docs_bm25, qa_index, qa_metadata, qa_bm25 = load_kb()

# =====================================
# SELF LEARNING
# =====================================

LEARN_FILE = "learned_examples.pkl"

if os.path.exists(LEARN_FILE):
    with open(LEARN_FILE, "rb") as f:
        learned_examples = pickle.load(f)
else:
    learned_examples = []

def store_learning(question, answer):
    if len(answer) > 200:
        existing = [e["q"].lower().strip() for e in learned_examples]
        if question.lower().strip() not in existing:
            learned_examples.append({"q": question, "a": answer})
            with open(LEARN_FILE, "wb") as f:
                pickle.dump(learned_examples, f)

def search_learned(query, top_k=2):
    if not learned_examples:
        return ""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ex in learned_examples:
        ex_tokens = set(re.findall(r"\w+", ex["q"].lower()))
        overlap = len(query_tokens & ex_tokens) / max(len(query_tokens), 1)
        if overlap > 0.3:
            scored.append((overlap, f"Q: {ex['q']}\nA: {ex['a']}"))
    scored.sort(reverse=True)
    return "\n\n".join(t for _, t in scored[:top_k])

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_rewritten" not in st.session_state:
    st.session_state.last_rewritten = ""

# =====================================
# INTENT CLASSIFIER
# =====================================

def classify_intent(prompt):
    """
    Classify what the user wants so we respond appropriately.
    Returns one of:
      - 'write_pql'     : user wants a PQL query written
      - 'explain_pql'   : user wants to understand a PQL function/concept
      - 'business_process' : user describes a business problem, wants PQL help
      - 'general'       : general Celonis / process mining question
    """
    p = prompt.lower()

    write_signals = [
        "write", "create", "generate", "build", "give me",
        "pql for", "query for", "how to calculate", "how do i calculate",
        "compute", "get the value", "find the", "show me the query"
    ]

    explain_signals = [
        "what is", "explain", "how does", "what does", "difference between",
        "when to use", "why use", "what are", "definition", "syntax of"
    ]

    business_signals = [
        "order to cash", "purchase to pay", "procure to pay", "p2p", "o2c",
        "lead time", "cycle time", "throughput time", "rework", "bottleneck",
        "conformance", "deviation", "invoice", "shipment", "delivery",
        "supplier", "customer", "vendor", "payment", "approval",
        "manufacturing", "finance", "supply chain", "procurement",
        "accounts payable", "accounts receivable", "sla", "kpi", "metric"
    ]

    if any(s in p for s in write_signals):
        return "write_pql"
    if any(s in p for s in explain_signals):
        return "explain_pql"
    if any(s in p for s in business_signals):
        return "business_process"
    return "general"

# =====================================
# PQL FUNCTION DETECTION
# =====================================

def detect_function(prompt):
    functions = [
        "PU_FIRST", "PU_LAST", "PU_AVG", "PU_SUM", "PU_COUNT",
        "PU_MIN", "PU_MAX", "PU_COUNT_DISTINCT",
        "DATEDIFF", "RUNNING_SUM", "COUNT_TABLE",
        "MOVING_AVG", "REMAP", "SOURCE", "TARGET",
        "RUNNING_TOTAL", "AVG", "MEDIAN", "FILTER"
    ]
    prompt_upper = prompt.upper()
    for f in functions:
        if re.search(rf"\b{f}\b", prompt_upper):
            return f
    return None

# =====================================
# FUNCTION ROUTING
# =====================================

def function_routing(f):
    normalized = f.lower().replace("_", "-")
    for item in docs_metadata:
        url = item.get("url", "").lower()
        if normalized in url or f.lower() in url:
            return item["text"]
    return None

# =====================================
# QUERY REWRITING
# =====================================

def rewrite_query(prompt):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"""Rewrite this into Celonis PQL search terminology.
Use PQL function names and process mining terms.
Return ONLY the rewritten query. No explanation.

Question: {prompt}"""
            }],
            temperature=0,
            max_tokens=120
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return prompt

# =====================================
# MULTI-QUERY GENERATION
# =====================================

def generate_queries(q):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"""Generate 3 alternative Celonis documentation search queries.
Each should approach the topic differently.
Return one query per line, no numbering.

Original: {q}"""
            }],
            temperature=0.3,
            max_tokens=150
        )
        lines = r.choices[0].message.content.strip().split("\n")
        return [l.strip() for l in lines if l.strip()][:3]
    except Exception:
        return []

# =====================================
# HYBRID SEARCH — BOTH STORES
# =====================================

def hybrid_search(query, top_k=5):
    emb    = np.array(EMBED_MODEL.encode([query])).astype("float32")
    tokens = re.findall(r"\w+", query.lower())
    results = []

    # Docs: Semantic
    D, I = docs_index.search(emb, top_k)
    for dist, idx in zip(D[0], I[0]):
        if idx < len(docs_metadata):
            results.append({"text": docs_metadata[idx]["text"], "score": (1/(1+float(dist))) * 0.6})

    # Docs: BM25
    bm = docs_bm25.get_scores(tokens)
    top_bm = np.argsort(bm)[::-1][:top_k]
    max_bm = bm[top_bm[0]] if bm[top_bm[0]] > 0 else 1
    for i in top_bm:
        if bm[i] > 0:
            results.append({"text": docs_metadata[i]["text"], "score": (bm[i]/max_bm) * 0.4})

    # QA: Semantic
    D2, I2 = qa_index.search(emb, top_k)
    for dist, idx in zip(D2[0], I2[0]):
        if idx < len(qa_metadata):
            results.append({"text": qa_metadata[idx], "score": (1/(1+float(dist))) * 0.6})

    # QA: BM25
    bm2 = qa_bm25.get_scores(tokens)
    top_bm2 = np.argsort(bm2)[::-1][:top_k]
    max_bm2 = bm2[top_bm2[0]] if bm2[top_bm2[0]] > 0 else 1
    for i in top_bm2:
        if bm2[i] > 0:
            results.append({"text": qa_metadata[i], "score": (bm2[i]/max_bm2) * 0.4})

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    return [r["text"] for r in ranked[:top_k]]

# =====================================
# DEDUPLICATION
# =====================================

def deduplicate(docs):
    seen, unique = set(), []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

# =====================================
# RERANKING
# =====================================

def rerank(query, docs, top_k=4):
    if not docs:
        return []
    pairs  = [[query, d] for d in docs]
    scores = RERANK_MODEL.predict(pairs)
    ranked = [d for _, d in sorted(zip(scores, docs), reverse=True)]
    return ranked[:top_k]

# =====================================
# RETRIEVE REFERENCE CONTEXT
# =====================================

def retrieve_reference(prompt):
    """
    Fetches relevant docs/examples from the knowledge base.
    This is used as ENHANCEMENT to LLM knowledge, not a strict constraint.
    """
    # Try exact function routing first
    func = detect_function(prompt)
    if func:
        routed = function_routing(func)
        if routed:
            st.session_state.last_rewritten = f"[Direct routing → {func}]"
            return routed

    # Rewrite + multi-query + hybrid search + rerank
    rewritten = rewrite_query(prompt)
    st.session_state.last_rewritten = rewritten

    all_queries = [rewritten] + generate_queries(rewritten)
    all_docs = []
    for q in all_queries:
        all_docs += hybrid_search(q, top_k=4)

    all_docs = deduplicate(all_docs)
    top_docs = rerank(rewritten, all_docs, top_k=4)

    learned = search_learned(prompt)

    parts = [d[:1200] for d in top_docs]
    if learned:
        parts.append("### From Previous Sessions:\n" + learned)

    return "\n\n".join(parts)[:6000]

# =====================================
# SYSTEM PROMPT — LLM-FIRST APPROACH
# =====================================

# This is the key change: LLM is the expert, docs are reference material.
# The model is free to answer using its own knowledge AND enhance with docs.

SYSTEM_PROMPT = """
You are an expert Celonis Process Mining Consultant and PQL (Process Query Language) specialist.

You have deep knowledge of:
- Celonis platform and its capabilities
- PQL syntax, functions, and best practices
- Business processes: Order to Cash, Procure to Pay, Accounts Payable,
  Accounts Receivable, Manufacturing, Supply Chain, Finance
- Process mining concepts: case tables, activity tables, event logs,
  throughput time, conformance checking, rework, bottlenecks, variants

YOUR BEHAVIOR:

1. ANSWERING QUESTIONS:
   - Answer confidently using your expert knowledge
   - Use the "Reference Material" provided to enhance and verify your answers
   - If reference material has exact syntax, always use it precisely
   - Do NOT say "not found in documentation" when you know the answer from expertise

2. WRITING PQL QUERIES:
   - When user asks for a PQL query, ALWAYS write one
   - Use official PQL syntax: functions like PU_AVG, DATEDIFF, COUNT_TABLE etc.
   - Always wrap table names and column names in double quotes: "TableName"."ColumnName"
   - Format queries clearly in a code block
   - Explain what each part does after the query

3. BUSINESS PROCESS QUESTIONS:
   - Map the business problem to the correct PQL approach
   - Example: "throughput time" → DATEDIFF between first and last activity
   - Suggest the right tables and columns for common process types

4. IMPORTANT PQL RULES:
   - PQL is NOT SQL — never use SELECT, FROM, WHERE, JOIN
   - Functions work on event log data: case tables and activity tables
   - Pull-up functions (PU_*) aggregate from a lower-grain to higher-grain table
   - FILTER is a PQL keyword, not a SQL WHERE clause
   - Column references always: "TABLE"."COLUMN"

5. TONE:
   - Be helpful, practical, and clear
   - Give working examples whenever possible
   - For complex queries, explain step by step
"""

# =====================================
# BUILD FINAL PROMPT BY INTENT
# =====================================

def build_prompt(prompt, intent, reference):
    """
    Builds the final user message based on classified intent.
    Reference docs always included but framed as enhancement, not constraint.
    """

    base = f"User Question:\n{prompt}\n\n"

    ref_block = ""
    if reference.strip():
        ref_block = f"""Reference Material from Celonis Documentation:
(Use this to verify syntax and enhance your answer.
You are NOT limited to only this material — use your expert knowledge too.)
---
{reference}
---

"""

    if intent == "write_pql":
        instruction = """Task: Write a PQL query that solves the user's request.
- Provide the complete PQL query in a code block
- Explain what each function/part does
- Mention which tables/columns are typically used for this in Celonis
- If multiple approaches exist, show the best one and mention alternatives
"""

    elif intent == "explain_pql":
        instruction = """Task: Explain the concept/function clearly.
- Give a clear definition
- Show the syntax with a practical example
- Explain when and why to use it
- Show a real business scenario where it applies
"""

    elif intent == "business_process":
        instruction = """Task: Help the user with their business process question.
- Identify the process type (O2C, P2P, Manufacturing etc.)
- Map their business question to the right PQL approach
- Write the PQL query if needed
- Explain which Celonis tables/columns are typically relevant
"""

    else:  # general
        instruction = """Task: Answer the question using your Celonis expertise.
- Be direct and practical
- Include PQL examples if relevant
- Connect to real business use cases
"""

    return base + ref_block + instruction

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## 🧠 Celonis PQL Copilot")
    st.markdown("---")

    st.markdown("**What I can help with:**")
    st.markdown("- ✍️ Write PQL queries")
    st.markdown("- 📖 Explain PQL functions")
    st.markdown("- 🏭 Business process analysis")
    st.markdown("- 🔍 Process mining concepts")
    st.markdown("---")

    st.markdown("**Knowledge Sources:**")
    st.markdown(f"- 📚 Doc chunks: `{len(docs_metadata)}`")
    st.markdown(f"- 💡 PQL examples: `{len(qa_metadata)}`")
    st.markdown(f"- 🧠 Learned Q&As: `{len(learned_examples)}`")
    st.markdown("---")

    st.markdown("**Try asking:**")
    st.caption("Write PQL to calculate throughput time in O2C")
    st.caption("Explain PU_AVG with a real example")
    st.caption("How do I find rework cases in manufacturing?")
    st.caption("What is DATEDIFF and when should I use it?")
    st.markdown("---")

    if st.session_state.last_rewritten:
        st.markdown("**🔍 Search Query Used:**")
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

st.title("🧠 Celonis PQL Copilot")
st.markdown(
    "Your expert assistant for **PQL queries**, **process mining**, "
    "and **Celonis** — ask anything."
)

# =====================================
# CHAT HISTORY
# =====================================

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask anything about Celonis, PQL, or your business process..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Classify intent
    intent = classify_intent(prompt)

    # Always retrieve reference material to enhance the answer
    reference = retrieve_reference(prompt)

    # Build the final prompt based on intent
    final_prompt = build_prompt(prompt, intent, reference)

    # Build API message history
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[:-1]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": final_prompt})

    answer = ""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=api_messages,
                    temperature=0.2,
                    max_tokens=1800
                )
                answer = r.choices[0].message.content
                st.markdown(answer)

            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Store good answers for future enhancement
    if answer and not answer.startswith("⚠️"):
        store_learning(prompt, answer)
