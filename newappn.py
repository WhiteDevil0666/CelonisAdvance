import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# BACKGROUND + FULL CSS FIX
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_style = f"""
        background:
            linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
            url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        """
    except FileNotFoundError:
        bg_style = "background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);"

    page_bg = f"""
    <style>

    /* ── App background ── */
    .stApp {{
        {bg_style}
    }}
    .main {{ background-color: transparent !important; }}
    section.main > div {{ background-color: transparent !important; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 0rem; }}

    /* ── Global text ── */
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6,
    p, div, span, label, li, td, th {{
        color: #e8e8f0 !important;
    }}

    /* ── Chat bubbles ── */
    .stChatMessage {{
        background-color: rgba(20, 20, 35, 0.88) !important;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px;
    }}

    /* ══════════════════════════════════════
       CODE BLOCKS — FULL DARK FIX
       ══════════════════════════════════════ */

    /* Inline code */
    code {{
        background-color: #1e1e2e !important;
        color: #cdd6f4 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    }}

    /* Fenced code block outer wrapper */
    .stCodeBlock,
    div[data-testid="stCode"],
    div[data-testid="stMarkdownContainer"] pre,
    .stMarkdown pre,
    pre {{
        background-color: #1e1e2e !important;
        border: 1px solid rgba(120, 120, 220, 0.35) !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
    }}

    /* Code text inside fenced block */
    pre code,
    .stCodeBlock code,
    div[data-testid="stCode"] code,
    div[data-testid="stMarkdownContainer"] pre code {{
        background-color: transparent !important;
        color: #cdd6f4 !important;
        font-size: 0.87rem !important;
        white-space: pre !important;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    }}

    /* Syntax highlight tokens — keep readable on dark bg */
    .stCodeBlock .keyword  {{ color: #cba6f7 !important; }}
    .stCodeBlock .string   {{ color: #a6e3a1 !important; }}
    .stCodeBlock .number   {{ color: #fab387 !important; }}
    .stCodeBlock .comment  {{ color: #6c7086 !important; font-style: italic; }}

    /* Copy button */
    .stCodeBlock button,
    div[data-testid="stCode"] button {{
        background-color: rgba(100, 100, 200, 0.35) !important;
        color: #cdd6f4 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }}

    /* ── Chat input ── */
    div[data-testid="stChatInput"] {{
        background-color: rgba(20, 20, 30, 0.95) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: #e8e8f0 !important;
        background-color: transparent !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: rgba(12, 12, 22, 0.97) !important;
    }}
    section[data-testid="stSidebar"] button {{
        background-color: rgba(40, 40, 70, 0.8) !important;
        color: #e8e8f0 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 4px !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        background-color: rgba(80, 80, 160, 0.6) !important;
    }}

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] > div {{
        background-color: rgba(30, 30, 50, 0.9) !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
    }}

    /* ── Bold text accent color ── */
    strong {{ color: #a0a0ff !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: #0f0f1a; }}
    ::-webkit-scrollbar-thumb {{ background: #3a3a6a; border-radius: 3px; }}

    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI with deep expertise in PQL (Process Query Language).

CORE CAPABILITIES:

1. ANSWER process mining conceptual questions clearly with real examples.
2. EXPLAIN any PQL function using provided documentation context.
3. WRITE CUSTOM PQL QUERIES when the user asks — this is a KEY capability.
   - Understand the user's business requirement.
   - Map it to correct PQL syntax.
   - Use Pull-Up functions (PU_AVG, PU_COUNT, PU_MAX, PU_MIN, PU_SUM, PU_MEDIAN, PU_LAST, PU_FIRST, etc.) correctly.
   - Apply FILTER, DOMAIN_TABLE, SOURCE, TARGET correctly.
   - Build multi-step KPI formulas if needed.
   - Always explain what the query does after writing it.

STRICT PQL RULES:

- Always use PQL syntax — NEVER SQL syntax.
- Preserve exact PQL operators: FILTER, DOMAIN_TABLE, SOURCE, TARGET, RUNNING_SUM, etc.
- Use double quotes for column names: "TABLE"."COLUMN"
- Pull-Up functions format: PU_FUNCTION ( DOMAIN_TABLE, COLUMN, FILTER condition )
- For date differences use: DATEDIFF ( 'unit', start, end )
- Case expressions: CASE WHEN ... THEN ... ELSE ... END
- If documentation context is provided, follow it strictly.
- If not in docs but user asks to write a query, use your PQL expertise to write it correctly.
- Never use SQL keywords: SELECT, FROM, WHERE, GROUP BY, JOIN
- Never simplify PQL into SQL-like syntax
- Never invent function names

RESPONSE FORMAT FOR QUERIES:

When writing a PQL query, always structure your response like this:

**PQL Query:**
```
<your PQL code here>
```

**Explanation:**
<step-by-step explanation of what the query does>

**Usage Note:**
<any important tips or variations>

PRIORITY: Technical Accuracy > Completeness > Simplicity
"""

# =====================================
# LOAD VECTOR STORE
# =====================================

@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index("pql_faiss.index")
        with open("pql_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.warning(f"⚠️ Vector store not loaded: {e}. Running in LLM-only mode.")
        return None, []

index, metadata = load_vector_store()

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Auto"

# =====================================
# INTENT DETECTION
# =====================================

def detect_intent(prompt):
    prompt_lower = prompt.lower()

    write_keywords = [
        "write", "generate", "create", "build", "give me", "make",
        "how to write", "query for", "pql for", "calculate", "compute",
        "show me query", "example query", "can you write", "can you create"
    ]
    pql_keywords = [
        "celonis", "pql", "pu_", "datediff", "process query",
        "pull-up", "throughput", "event log", "case duration",
        "activity", "variant", "process", "kpi", "filter", "domain_table",
        "rework", "bottleneck", "cycle time", "lead time"
    ]

    is_write_request = any(w in prompt_lower for w in write_keywords)
    is_pql_topic = any(w in prompt_lower for w in pql_keywords)

    if is_write_request and is_pql_topic:
        return "write_pql"
    elif is_pql_topic:
        return "explain_pql"
    else:
        return "general"

# =====================================
# EXACT FUNCTION ROUTING
# =====================================

def exact_function_match(query):
    if not metadata:
        return None
    query_upper = query.upper()
    tokens = re.findall(r'\bPU_[A-Z_]+\b', query_upper)
    for token in tokens:
        for item in metadata:
            url = item.get("url", "").lower()
            if token.lower() in url:
                return item["text"]
    return None

# =====================================
# SEMANTIC SEARCH
# =====================================

def semantic_search(query, top_k=8):
    if index is None or not metadata:
        return ""
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx]["text"])
    return "\n\n---\n\n".join(results)

# =====================================
# CONTEXT PIPELINE
# =====================================

def retrieve_context(prompt):
    exact_match = exact_function_match(prompt)
    if exact_match:
        return f"[EXACT MATCH FOUND]\n\n{exact_match}"
    return semantic_search(prompt)

# =====================================
# BUILD FINAL PROMPT
# =====================================

def build_final_prompt(prompt, intent):
    if intent == "general":
        return prompt

    context = retrieve_context(prompt)

    if intent == "write_pql":
        if context.strip():
            return f"""
TASK: Write a custom PQL query based on the user's requirement.

Documentation Context (use as reference for syntax):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Requirement:
{prompt}

Instructions:
- Write a complete, working PQL query.
- Follow exact PQL syntax from the documentation above.
- Explain the query step by step after writing it.
"""
        else:
            return f"""
TASK: Write a custom PQL query based on the user's requirement.

No documentation context found. Use your expert PQL knowledge.

User Requirement:
{prompt}

Instructions:
- Write a complete, working PQL query using correct PQL syntax.
- Explain the query step by step after writing it.
"""

    elif intent == "explain_pql":
        if context.strip():
            return f"""
TASK: Explain the PQL concept or function the user is asking about.

Documentation Context:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Question:
{prompt}
"""
        else:
            return prompt

    return prompt

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.session_state.mode = st.selectbox(
        "Response Mode",
        ["Auto", "Always use docs", "LLM only"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 💬 Chat")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📘 Quick PQL Examples")
    examples = [
        "Write a PQL query for average case duration",
        "How does PU_AVG work in Celonis?",
        "Write PQL to count cases per variant",
        "PQL query to find rework activities",
        "Calculate throughput time between two activities",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.markdown("---")
    index_status = "✅ Loaded" if index is not None else "⚠️ LLM-only mode"
    st.markdown(f"**Vector Index:** {index_status}")
    st.markdown(f"**Model:** `{MODEL_NAME}`")

# =====================================
# HEADER
# =====================================

st.title("🧠 Celonis Process Mining Copilot")
st.markdown("*Powered by Divyansh · Ask anything about PQL or Process Mining*")
st.markdown("---")

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask a question or say: 'Write a PQL query for...'"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = detect_intent(prompt)

    if st.session_state.mode == "Always use docs":
        intent = "explain_pql" if intent == "general" else intent
    elif st.session_state.mode == "LLM only":
        intent = "general"

    final_prompt = build_final_prompt(prompt, intent)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"❌ Error calling Groq API: `{str(e)}`"

            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
