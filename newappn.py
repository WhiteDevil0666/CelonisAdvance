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
# INTENT + SCOPE CLASSIFIER
# =====================================

# FIX 1: Added out-of-scope detection so non-PQL questions
# (like UI setup, connections, admin tasks) don't get polluted
# with irrelevant PQL context injected from the knowledge base.

def classify_intent(prompt):
    """
    Returns a tuple: (intent, is_pql_related)

    intent options:
      - 'write_pql'        : user wants a PQL query written
      - 'explain_pql'      : user wants to understand PQL function/concept
      - 'business_process' : business problem needing PQL solution
      - 'platform_help'    : Celonis UI, setup, connectors, admin — NO PQL injection
      - 'general'          : general process mining / Celonis question

    is_pql_related:
      - True  → retrieve docs/examples from knowledge base to enhance answer
      - False → answer directly, do NOT inject PQL context (prevents confusion)
    """
    p = prompt.lower()

    # --- Out-of-scope: Platform/Admin/Setup questions ---
    # These should be answered cleanly without PQL context injection
    platform_signals = [
        "connection", "connect", "on-premise", "on premise", "connector",
        "install", "installation", "setup", "configure", "configuration",
        "login", "sign in", "account", "permission", "role", "user management",
        "data pool", "data job", "transformation", "extractor",
        "upload", "import", "export", "download",
        "celonis studio", "process hub", "action flow", "app",
        "notification", "alert", "schedule", "automation",
        "sap extractor", "jdbc", "api key", "token", "oauth",
        "how to access", "where do i", "how do i open", "navigate",
        "dashboard", "workspace", "tenant", "deployment"
    ]
    if any(s in p for s in platform_signals):
        return "platform_help", False

    # --- Write PQL ---
    write_signals = [
        "write", "create", "generate", "build", "give me a query",
        "pql for", "query for", "how to calculate", "how do i calculate",
        "compute", "get the value of", "find the", "show me the query",
        "calculate the", "what pql"
    ]
    if any(s in p for s in write_signals):
        return "write_pql", True

    # --- Explain PQL ---
    explain_signals = [
        "what is", "explain", "how does", "what does", "difference between",
        "when to use", "why use", "what are", "definition of", "syntax of",
        "how to use"
    ]
    if any(s in p for s in explain_signals):
        return "explain_pql", True

    # --- Business process ---
    business_signals = [
        "order to cash", "purchase to pay", "procure to pay", "p2p", "o2c",
        "lead time", "cycle time", "throughput time", "rework", "bottleneck",
        "conformance", "deviation", "invoice", "shipment", "delivery",
        "supplier", "customer", "vendor", "payment", "approval",
        "manufacturing", "finance", "supply chain", "procurement",
        "accounts payable", "accounts receivable", "sla", "kpi", "metric",
        "process variant", "happy path", "case duration"
    ]
    if any(s in p for s in business_signals):
        return "business_process", True

    return "general", True

# =====================================
# PQL FUNCTION DETECTION
# =====================================

def detect_function(prompt):
    functions = [
        "PU_FIRST", "PU_LAST", "PU_AVG", "PU_SUM", "PU_COUNT",
        "PU_MIN", "PU_MAX", "PU_COUNT_DISTINCT", "PU_STRING_AGG",
        "PU_MEDIAN", "PU_VAR", "PU_STDEV",
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
            results.append({"text": docs_metadata[idx]["text"],
                            "score": (1 / (1 + float(dist))) * 0.6})

    # Docs: BM25
    bm = docs_bm25.get_scores(tokens)
    top_bm = np.argsort(bm)[::-1][:top_k]
    max_bm = bm[top_bm[0]] if bm[top_bm[0]] > 0 else 1
    for i in top_bm:
        if bm[i] > 0:
            results.append({"text": docs_metadata[i]["text"],
                            "score": (bm[i] / max_bm) * 0.4})

    # QA: Semantic
    D2, I2 = qa_index.search(emb, top_k)
    for dist, idx in zip(D2[0], I2[0]):
        if idx < len(qa_metadata):
            results.append({"text": qa_metadata[idx],
                            "score": (1 / (1 + float(dist))) * 0.6})

    # QA: BM25
    bm2 = qa_bm25.get_scores(tokens)
    top_bm2 = np.argsort(bm2)[::-1][:top_k]
    max_bm2 = bm2[top_bm2[0]] if bm2[top_bm2[0]] > 0 else 1
    for i in top_bm2:
        if bm2[i] > 0:
            results.append({"text": qa_metadata[i],
                            "score": (bm2[i] / max_bm2) * 0.4})

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
    """Only called when is_pql_related=True."""
    func = detect_function(prompt)
    if func:
        routed = function_routing(func)
        if routed:
            st.session_state.last_rewritten = f"[Direct routing → {func}]"
            return routed

    rewritten = rewrite_query(prompt)
    st.session_state.last_rewritten = rewritten

    all_queries = [rewritten] + generate_queries(rewritten)
    all_docs = []
    for q in all_queries:
        all_docs += hybrid_search(q, top_k=4)

    all_docs  = deduplicate(all_docs)
    top_docs  = rerank(rewritten, all_docs, top_k=4)
    learned   = search_learned(prompt)

    parts = [d[:1200] for d in top_docs]
    if learned:
        parts.append("### From Previous Sessions:\n" + learned)

    return "\n\n".join(parts)[:6000]

# =====================================
# SYSTEM PROMPTS
# =====================================

# FIX 2 & 3: Tighter PQL syntax rules + self-verification instruction
# so the model checks its own explanation matches the query it wrote.

PQL_SYSTEM_PROMPT = """
You are an expert Celonis Process Mining Consultant and PQL specialist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT PQL SYNTAX RULES — NEVER VIOLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PQL is NOT SQL. Never use: SELECT, FROM, WHERE, JOIN, GROUP BY, HAVING.

2. Always quote table and column names with double quotes:
   ✅ "TableName"."ColumnName"
   ❌ TableName.ColumnName

3. Pull-Up function syntax (PU_*):
   PU_FUNCTION ( target_table, source_table.column [, filter] [ORDER BY col ASC|DESC] )
   Example:
   PU_AVG ( "CaseTable", "ActivityTable"."Duration" )
   PU_FIRST ( "CaseTable", "ActivityTable"."Activity", ORDER BY "ActivityTable"."Eventtime" ASC )

4. DATEDIFF syntax:
   DATEDIFF ( time_unit, "Table"."StartColumn", "Table"."EndColumn" )
   Example:
   DATEDIFF ( HOURS, "Cases"."CreateDate", "Cases"."CloseDate" )

5. FILTER syntax (PQL FILTER, not SQL WHERE):
   FILTER "Table"."Column" = 'value'

6. COUNT_TABLE syntax:
   COUNT_TABLE ( "TableName" )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-VERIFICATION RULE — ALWAYS APPLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After writing ANY PQL query, before giving the explanation:
- Re-read your query carefully
- Check each argument in the function matches what you explain
- Check argument order matches official syntax
- If your explanation says "argument 2 is X", verify argument 2 in the
  query IS actually X — fix if wrong
- Never describe an argument as something it is not

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Answer confidently using expert knowledge + reference material
- Always write PQL when asked — never refuse
- If reference material has exact syntax, use it precisely
- Map business problems to correct PQL approach
- Format every PQL query in a code block
- Explain what each argument does AFTER verifying query is correct
- Keep explanations concise and practical
"""

# FIX 1: Separate clean prompt for platform/admin questions
# No PQL context injected — prevents mixing UI answers with random PQL
PLATFORM_SYSTEM_PROMPT = """
You are a Celonis platform expert.

Answer questions about Celonis platform setup, configuration, connections,
data jobs, extractors, Studio, Process Hub, and administration clearly.

Rules:
- Give step-by-step guidance where appropriate
- Be practical and direct
- Do NOT include PQL queries unless the user specifically asks for them
- If you are unsure about a specific UI step that may have changed,
  advise the user to check the official Celonis documentation at:
  https://docs.celonis.com
"""

# =====================================
# BUILD FINAL PROMPT BY INTENT
# =====================================

def build_prompt(prompt, intent, reference):
    base    = f"User Question:\n{prompt}\n\n"
    ref_block = ""

    if reference.strip():
        ref_block = f"""Reference Material from Celonis Documentation:
(Use this to verify and enhance your answer.
You are NOT limited to only this material — use your expert knowledge too.)
---
{reference}
---

"""

    instructions = {
        "write_pql": """Task: Write the PQL query that solves the user's request.
1. Write the complete PQL query in a code block
2. Self-verify: re-read the query and confirm every argument is correct
3. Explain each argument accurately (must match the query exactly)
4. Mention which tables/columns are typically used for this in Celonis
5. If multiple approaches exist, show the best one
""",
        "explain_pql": """Task: Explain the PQL concept or function clearly.
1. Give the official syntax
2. Explain each argument precisely
3. Show a working example in a code block
4. Self-verify the example is correct before presenting it
5. Describe a real business scenario where this applies
""",
        "business_process": """Task: Help the user solve their business process challenge.
1. Identify the process type (O2C, P2P, AP, Manufacturing, etc.)
2. Map the business question to the right PQL approach
3. Write the PQL query in a code block
4. Self-verify the query is syntactically correct
5. Explain which tables/columns are typically relevant in Celonis
""",
        "general": """Task: Answer the Celonis/process mining question.
1. Be direct and practical
2. Include PQL examples if relevant (in code blocks)
3. Connect to real business use cases
"""
    }

    instruction = instructions.get(intent, instructions["general"])
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
    st.markdown("- ⚙️ Celonis platform questions")
    st.markdown("---")
    st.markdown("**Knowledge Base:**")
    st.markdown(f"- 📚 Doc chunks: `{len(docs_metadata)}`")
    st.markdown(f"- 💡 PQL examples: `{len(qa_metadata)}`")
    st.markdown(f"- 🧠 Learned Q&As: `{len(learned_examples)}`")
    st.markdown("---")
    st.markdown("**Try asking:**")
    st.caption("Write PQL to calculate throughput time in O2C")
    st.caption("Explain PU_FIRST with a real example")
    st.caption("How do I find rework cases in manufacturing?")
    st.caption("What is DATEDIFF and when should I use it?")
    st.caption("How do I connect SAP to Celonis?")
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

    # Step 1: Classify intent and whether PQL context is needed
    intent, is_pql_related = classify_intent(prompt)

    # Step 2: Choose system prompt based on scope
    # FIX 1: Platform questions get clean prompt, no PQL docs injected
    system_prompt = PLATFORM_SYSTEM_PROMPT if intent == "platform_help" else PQL_SYSTEM_PROMPT

    # Step 3: Only retrieve knowledge base context for PQL-related questions
    reference = ""
    if is_pql_related:
        reference = retrieve_reference(prompt)
    else:
        # No retrieval for platform/admin — just answer directly and cleanly
        st.session_state.last_rewritten = "[Platform question — no PQL context needed]"

    # Step 4: Build the final user prompt
    if intent == "platform_help":
        final_prompt = f"Question:\n{prompt}"
    else:
        final_prompt = build_prompt(prompt, intent, reference)

    # Step 5: Build API message history
    api_messages = [{"role": "system", "content": system_prompt}]
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

    if answer and not answer.startswith("⚠️"):
        store_learning(prompt, answer)
