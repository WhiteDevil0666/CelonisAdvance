import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64
import csv
import os
from datetime import datetime

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Celonis Process Mining Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# CSS — ALL WHITE AREAS FIXED
# ════════════════════════════════════════════════════════════

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_style = f"""
            background:
                linear-gradient(rgba(0,0,0,0.83), rgba(0,0,0,0.83)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        """
    except FileNotFoundError:
        bg_style = "background: linear-gradient(135deg, #0d0d1a 0%, #141428 50%, #0f1a2e 100%);"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── App & base ── */
    .stApp {{ {bg_style} }}
    .main, section.main > div, .block-container {{
        background-color: transparent !important;
    }}
    .block-container {{ padding-top: 2rem !important; padding-bottom: 1rem !important; }}
    html, body {{ background-color: #0d0d1a !important; color: #e2e2f0 !important; }}

    /* ── Text ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown span, .stMarkdown div,
    p, span, label, li, td, th {{ color: #e2e2f0 !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: #ffffff !important; }}
    strong, b {{ color: #a78bfa !important; }}
    em {{ color: #94a3b8 !important; }}

    /* ── Chat messages ── */
    .stChatMessage {{
        background-color: rgba(18, 18, 32, 0.92) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 8px !important;
    }}

    /* ── Code blocks ── */
    pre {{
        background-color: #13131f !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 10px !important;
        padding: 16px !important;
        overflow-x: auto !important;
    }}
    pre code {{
        background-color: transparent !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.85rem !important;
        white-space: pre !important;
        line-height: 1.6 !important;
    }}
    code {{
        background-color: #1e1e35 !important;
        color: #c9d1d9 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.85rem !important;
    }}
    .stCodeBlock, div[data-testid="stCode"] {{
        background-color: #13131f !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 10px !important;
    }}
    .stCodeBlock code, div[data-testid="stCode"] code {{
        background-color: transparent !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ── Chat input — nuclear white fix ── */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div,
    div[data-testid="stChatInput"] > div > div > div {{
        background-color: #13131f !important;
        border-radius: 14px !important;
    }}
    div[data-testid="stChatInput"] {{
        border: 1px solid rgba(139,92,246,0.45) !important;
    }}
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus,
    div[data-testid="stChatInput"] textarea:active,
    div[data-testid="stChatInput"] textarea:hover {{
        background-color: #13131f !important;
        background: #13131f !important;
        color: #e2e2f0 !important;
        caret-color: #a78bfa !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #6060aa !important;
        opacity: 1 !important;
    }}
    div[data-testid="stChatInput"] button,
    div[data-testid="stChatInput"] button:hover {{
        background-color: rgba(139,92,246,0.35) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: #0b0b18 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }}
    section[data-testid="stSidebar"] * {{ color: #e2e2f0 !important; }}
    section[data-testid="stSidebar"] .stButton button {{
        background-color: rgba(30, 30, 55, 0.9) !important;
        color: #e2e2f0 !important;
        border: 1px solid rgba(139,92,246,0.25) !important;
        border-radius: 9px !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 5px !important;
        font-size: 0.82rem !important;
        padding: 8px 12px !important;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background-color: rgba(139,92,246,0.25) !important;
        border-color: rgba(139,92,246,0.6) !important;
    }}

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] > div > div {{
        background-color: #1a1a2e !important;
        color: #e2e2f0 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 8px !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: #1a1a2e !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] li {{ color: #e2e2f0 !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{
        background-color: rgba(139,92,246,0.2) !important;
    }}

    /* ── Alert boxes ── */
    div[data-testid="stAlert"] {{
        background-color: rgba(30,30,60,0.8) !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        color: #e2e2f0 !important;
        border-radius: 8px !important;
    }}

    /* ── Divider + scrollbar ── */
    hr {{ border-color: rgba(255,255,255,0.08) !important; }}
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: #0d0d1a; }}
    ::-webkit-scrollbar-thumb {{ background: #3a3a6a; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #6060aa; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background("background.png")

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════

MODEL_NAME  = "llama-3.3-70b-versatile"
client      = Groq(api_key=st.secrets["GROQ_API_KEY"])
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Full list of valid PQL functions for reference
VALID_PQL_FUNCTIONS = [
    "PU_COUNT", "PU_SUM", "PU_AVG", "PU_MIN", "PU_MAX",
    "PU_FIRST", "PU_LAST", "PU_MEDIAN", "PU_MODE", "PU_STDEV",
    "PU_COUNT_DISTINCT", "PU_STRING_AGG", "PU_TRIMMED_MEAN",
    "PU_PRODUCT", "PU_QUANTILE",
    "DATEDIFF", "RUNNING_SUM", "RUNNING_TOTAL",
    "ACTIVATION_COUNT", "SOURCE", "TARGET",
    "CALC_THROUGHPUT", "CALC_REWORK",
    "MOVING_AVG", "MOVING_SUM", "MOVING_COUNT", "MOVING_MAX", "MOVING_MIN",
    "WINDOW_AVG", "INTERPOLATE",
    "CASE WHEN", "COALESCE", "ISNULL",
    "MATCH_PROCESS", "MATCH_ACTIVITIES", "VARIANT",
    "INDEX_ACTIVITY_ORDER", "INDEX_ACTIVITY_TYPE",
    "LAG", "LEAD", "ACTIVITY_LAG", "ACTIVITY_LEAD",
]

# ════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI with deep expertise in PQL.
You have a knowledge base of 195+ PQL functions with real solved examples.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ANSWER EVERY QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Understand the process mining task.
Step 2 — Identify which PQL function(s) are needed.
Step 3 — Study the Knowledge Base examples in the context.
Step 4 — Write syntactically correct, properly adapted PQL.
Step 5 — Explain it clearly, line by line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT PQL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Column reference:   "TABLE"."COLUMN"
✅ Pull-Up syntax:     PU_FUNCTION ( DOMAIN_TABLE ( "TABLE" ), "TABLE"."COL", FILTER ... )
✅ Date diff:          DATEDIFF ( 'day', "TABLE"."START", "TABLE"."END" )
✅ Case expression:    CASE WHEN condition THEN value ELSE other END
✅ Filter inside PU_:  FILTER "TABLE"."COL" = 'value'
✅ Source / Target:    SOURCE ( "TABLE"."ACTIVITY", EXCLUDE DUPLICATE ACTIVITIES )
✅ Running agg:        RUNNING_SUM ( "TABLE"."COLUMN" )
✅ Rework check:       ACTIVATION_COUNT ( "TABLE"."ACTIVITY" ) > 1

❌ Never use SQL: SELECT, FROM, WHERE, GROUP BY, JOIN
❌ Never invent function names
❌ Never use generic placeholders if user gave real names
❌ If user gave no names, use: <CASE_TABLE>, <ACTIVITY_COL>, <VALUE_COL>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — ALWAYS USE THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**📌 PQL Query:**
```
<properly indented PQL>
```

**📖 What this does:**
<plain English, line-by-line explanation>

**💡 Note:**
<tips, what to replace, edge cases>
"""

# ════════════════════════════════════════════════════════════
# VALIDATOR PROMPT
# ════════════════════════════════════════════════════════════

VALIDATOR_PROMPT = """
You are a strict Celonis PQL syntax validator.

Check the given PQL for:
  • Missing or extra commas
  • Wrong FILTER syntax
  • Invalid DOMAIN_TABLE usage
  • SQL keywords (SELECT, FROM, WHERE, GROUP BY, JOIN)
  • Wrong column format (must be "TABLE"."COLUMN")
  • Invented or invalid function names

Respond ONLY in this format:

**✅ Validation Status:** <PASSED or ISSUES FOUND>

**🔧 Corrected PQL Query:**
```
<corrected query, or original if no issues>
```

**📝 Notes:**
<list each fix, or "No issues found.">
"""

# ════════════════════════════════════════════════════════════
# LOAD VECTOR STORE
# ════════════════════════════════════════════════════════════

@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index("pql_faiss.index")
        with open("pql_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.warning(f"⚠️ Knowledge base not found ({e}). Running in LLM-only mode.")
        return None, []


vector_index, vector_metadata = load_vector_store()

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Auto"

if "validate" not in st.session_state:
    st.session_state.validate = True

# ════════════════════════════════════════════════════════════
# INTENT DETECTION
# ════════════════════════════════════════════════════════════

def detect_intent(prompt: str) -> str:
    p = prompt.lower()
    write_kw = [
        "write", "generate", "create", "build", "give me", "make a",
        "query for", "pql for", "calculate", "compute", "show query",
        "can you write", "can you create", "help me write", "i need a query",
        "i want a query", "example query", "how would i", "how to write",
        "how do i write", "need pql"
    ]
    pql_kw = [
        "celonis", "pql", "pu_", "datediff", "pull-up", "throughput",
        "event log", "case duration", "activity", "variant", "kpi",
        "filter", "domain_table", "rework", "bottleneck", "cycle time",
        "lead time", "source", "target", "running_sum", "activation_count",
        "process mining", "process query", "avg", "count", "sum",
        "max", "min", "median", "moving", "window", "interpolate",
        "calc_throughput", "calc_rework", "timestamp", "first activity",
        "last activity", "index", "lag", "lead", "variant"
    ]
    is_write = any(w in p for w in write_kw)
    is_pql   = any(w in p for w in pql_kw)

    if is_write and is_pql:
        return "write_pql"
    elif is_pql:
        return "explain_pql"
    else:
        return "general"

# ════════════════════════════════════════════════════════════
# PATTERN ENGINE — instant templates for common requests
# ════════════════════════════════════════════════════════════

PATTERN_TEMPLATES = {
    "throughput": {
        "label": "Throughput / Case Duration",
        "query": '''DATEDIFF(
    'day',
    PU_MIN(
        DOMAIN_TABLE("<CASE_TABLE>"),
        "<EVENT_TABLE>"."TIMESTAMP"
    ),
    PU_MAX(
        DOMAIN_TABLE("<CASE_TABLE>"),
        "<EVENT_TABLE>"."TIMESTAMP"
    )
)'''
    },
    "rework": {
        "label": "Rework Detection",
        "query": '''ACTIVATION_COUNT(
    "<EVENT_TABLE>"."ACTIVITY"
) > 1'''
    },
    "sum": {
        "label": "Sum per Case",
        "query": '''PU_SUM(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."<VALUE_COL>"
)'''
    },
    "avg": {
        "label": "Average per Case",
        "query": '''PU_AVG(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."<VALUE_COL>"
)'''
    },
    "count": {
        "label": "Count Events per Case",
        "query": '''PU_COUNT(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."<EVENT_ID_COL>"
)'''
    },
    "first": {
        "label": "First Activity per Case",
        "query": '''PU_FIRST(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."ACTIVITY",
    ORDER BY "<EVENT_TABLE>"."TIMESTAMP" ASC
)'''
    },
    "last": {
        "label": "Last Activity per Case",
        "query": '''PU_LAST(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."ACTIVITY",
    ORDER BY "<EVENT_TABLE>"."TIMESTAMP" ASC
)'''
    },
    "case_when": {
        "label": "CASE WHEN Classification",
        "query": '''CASE
    WHEN DATEDIFF('day', "<CASE_TABLE>"."START", "<CASE_TABLE>"."END") <= 5
        THEN 'Fast'
    WHEN DATEDIFF('day', "<CASE_TABLE>"."START", "<CASE_TABLE>"."END") <= 15
        THEN 'Medium'
    ELSE 'Slow'
END'''
    },
    "running_sum": {
        "label": "Running Sum",
        "query": '''RUNNING_SUM(
    "<EVENT_TABLE>"."<VALUE_COL>"
)'''
    },
    "filter": {
        "label": "Filtered Aggregation",
        "query": '''PU_COUNT(
    DOMAIN_TABLE("<CASE_TABLE>"),
    "<EVENT_TABLE>"."<EVENT_ID_COL>",
    FILTER "<EVENT_TABLE>"."ACTIVITY" = '<ACTIVITY_NAME>'
)'''
    },
}


def detect_pattern(prompt: str) -> str | None:
    p = prompt.lower()
    if any(w in p for w in ["throughput", "time between", "cycle time", "lead time", "duration", "case duration"]):
        return "throughput"
    if "rework" in p or "repeated" in p or "loop" in p:
        return "rework"
    if any(w in p for w in ["running sum", "running total", "cumulative"]):
        return "running_sum"
    if any(w in p for w in ["first activity", "first event", "first step"]):
        return "first"
    if any(w in p for w in ["last activity", "last event", "last step", "current status"]):
        return "last"
    if any(w in p for w in ["classify", "classification", "categorize", "label", "case when"]):
        return "case_when"
    if any(w in p for w in ["filter", "only for", "where activity"]):
        return "filter"
    if any(w in p for w in ["total", "sum"]):
        return "sum"
    if any(w in p for w in ["average", "avg", "mean"]):
        return "avg"
    if "count" in p:
        return "count"
    return None


def get_pattern_query(pattern: str) -> str | None:
    tmpl = PATTERN_TEMPLATES.get(pattern)
    if tmpl:
        return tmpl["query"]
    return None

# ════════════════════════════════════════════════════════════
# RETRIEVAL — exact match + semantic search
# ════════════════════════════════════════════════════════════

def exact_function_match(query: str) -> list:
    if not vector_metadata:
        return []
    q_upper = query.upper()
    matches, seen = [], set()
    for item in vector_metadata:
        func = item.get("function", "").upper()
        if func and func in q_upper and func not in seen:
            matches.append(item)
            seen.add(func)
    return matches


def semantic_search(query: str, top_k: int = 6) -> list:
    if vector_index is None or not vector_metadata:
        return []
    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb, dtype="float32")
    faiss.normalize_L2(emb)
    _, I = vector_index.search(emb, top_k)
    return [vector_metadata[i] for i in I[0] if i < len(vector_metadata)]


def retrieve_context(prompt: str) -> str:
    exact    = exact_function_match(prompt)
    semantic = semantic_search(prompt, top_k=6)

    seen, combined = set(), []
    for item in exact + semantic:
        key = (item.get("function", ""), item.get("question", ""))
        if key not in seen:
            seen.add(key)
            combined.append(item)

    if not combined:
        return ""

    blocks = []
    for item in combined[:8]:
        blocks.append(
            f"Function: {item.get('function', '')}\n"
            f"Example Question: {item.get('question', '')}\n"
            f"PQL Answer:\n{item.get('answer', '')}"
        )
    return ("\n\n" + "─" * 50 + "\n\n").join(blocks)

# ════════════════════════════════════════════════════════════
# BUILD FINAL PROMPT
# ════════════════════════════════════════════════════════════

def build_final_prompt(prompt: str, intent: str) -> str:
    if intent == "general":
        return prompt

    context = retrieve_context(prompt)

    if context.strip():
        ctx_block = (
            f"Knowledge Base Examples (study syntax, adapt to user's requirement):\n"
            f"{'━' * 50}\n{context}\n{'━' * 50}\n\n"
        )
    else:
        ctx_block = (
            "No matching examples found in knowledge base. "
            "Use your expert PQL knowledge.\n\n"
        )

    functions_hint = (
        f"Available PQL functions for reference:\n"
        f"{', '.join(VALID_PQL_FUNCTIONS)}\n\n"
    )

    if intent == "write_pql":
        return (
            f"TASK: Write a custom PQL query for the user's exact requirement.\n\n"
            f"{ctx_block}"
            f"{functions_hint}"
            f"User Requirement:\n{prompt}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Study the syntax patterns from the Knowledge Base examples above.\n"
            f"- Adapt and combine them to match the user's specific requirement.\n"
            f"- Use the user's actual table/column names if they provided them.\n"
            f"- If no names given, use: <CASE_TABLE>, <EVENT_TABLE>, <ACTIVITY_COL>, <VALUE_COL>.\n"
            f"- Write valid PQL only — never SQL.\n"
            f"- Explain every line of the query after writing it.\n"
        )
    elif intent == "explain_pql":
        return (
            f"TASK: Explain the PQL function or concept the user is asking about.\n\n"
            f"{ctx_block}"
            f"User Question:\n{prompt}\n"
        )
    return prompt

# ════════════════════════════════════════════════════════════
# EXTRACT PQL FROM RESPONSE
# ════════════════════════════════════════════════════════════

def extract_pql(text: str) -> str:
    match = re.search(r"```.*?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ════════════════════════════════════════════════════════════
# VALIDATE PQL
# ════════════════════════════════════════════════════════════

def validate_pql(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user",   "content": f"Validate this PQL query:\n\n```\n{query}\n```"}
            ],
            temperature=0,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Validator error: {e}"

# ════════════════════════════════════════════════════════════
# SELF-LEARNING — SAVE TO CSV
# ════════════════════════════════════════════════════════════

def save_training_example(question: str, query: str):
    file   = "training_data.csv"
    exists = os.path.isfile(file)
    try:
        with open(file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["Timestamp", "Question", "PQL_Query"])
            writer.writerow([datetime.now().isoformat(), question, query])
    except Exception:
        pass

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.session_state.mode = st.selectbox(
        "Response Mode",
        ["Auto", "Always use Knowledge Base", "LLM only"],
        index=0,
        help=(
            "Auto = smart intent detection.\n"
            "Knowledge Base = always retrieve examples.\n"
            "LLM only = skip retrieval."
        )
    )

    st.session_state.validate = st.toggle(
        "🔎 Auto-validate PQL",
        value=True,
        help="Runs a second pass to check and fix PQL syntax errors."
    )

    st.markdown("---")
    st.markdown("### 💬 Chat")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Pattern Templates")
    st.markdown(
        "<small style='color:#6060aa'>Click to instantly load a template query</small>",
        unsafe_allow_html=True
    )
    pattern_labels = {
        "throughput":  "⏱ Throughput / Duration",
        "rework":      "🔁 Rework Detection",
        "avg":         "📊 Average per Case",
        "sum":         "➕ Sum per Case",
        "count":       "🔢 Count Events",
        "first":       "🥇 First Activity",
        "last":        "🏁 Last Activity",
        "case_when":   "🏷 Classify Cases",
        "running_sum": "📈 Running Sum",
        "filter":      "🔍 Filtered Aggregation",
    }
    for pat, label in pattern_labels.items():
        if st.button(label, key=f"pat_{pat}"):
            q = get_pattern_query(pat)
            msg = (
                f"**📌 PQL Template — {PATTERN_TEMPLATES[pat]['label']}:**\n\n"
                f"```\n{q}\n```\n\n"
                f"💡 Replace `<CASE_TABLE>`, `<EVENT_TABLE>`, `<VALUE_COL>` "
                f"with your actual table and column names."
            )
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.rerun()

    st.markdown("---")
    st.markdown("### 📘 Quick Examples")
    examples = [
        "Write PQL for average case duration",
        "How does PU_COUNT work?",
        "Write PQL to detect rework activities",
        "Calculate throughput time between two activities",
        "How to use DATEDIFF in Celonis?",
        "Write PQL using CASE WHEN for classification",
        "How does RUNNING_SUM work?",
        "Write PQL for first activity per case",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.markdown("---")
    kb_status = (
        f"✅ {len(vector_metadata)} chunks loaded"
        if vector_metadata
        else "⚠️ Not loaded (LLM-only mode)"
    )
    st.markdown(f"**Knowledge Base:** {kb_status}")
    st.markdown(f"**Model:** `{MODEL_NAME}`")

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════

st.title("🧠 Celonis Process Mining Copilot")
st.markdown(
    "*Powered by Divyansh · 195+ PQL functions · "
    "Ask anything or say: \"Write a PQL query for...\"*"
)
st.markdown("---")

# ════════════════════════════════════════════════════════════
# DISPLAY CHAT HISTORY
# ════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ════════════════════════════════════════════════════════════
# CHAT INPUT
# ════════════════════════════════════════════════════════════

if prompt := st.chat_input("Ask a PQL question or say: 'Write a PQL query for...'"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Determine intent ────────────────────────────────────
    intent = detect_intent(prompt)
    if st.session_state.mode == "Always use Knowledge Base":
        intent = "explain_pql" if intent == "general" else intent
    elif st.session_state.mode == "LLM only":
        intent = "general"

    # ── Check pattern engine first ──────────────────────────
    pattern       = detect_pattern(prompt)
    pattern_query = get_pattern_query(pattern) if pattern else None

    # Build conversation history (exclude latest — replaced by final_prompt)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & writing PQL..."):

            # ── Step 1: Generate answer ──────────────────────
            if pattern_query and intent in ("write_pql",):
                # Inject the pattern as a strong hint into the prompt
                hint = (
                    f"PATTERN HINT — Use this as base syntax and adapt it:\n"
                    f"```\n{pattern_query}\n```\n\n"
                )
                final_prompt = hint + build_final_prompt(prompt, intent)
            else:
                final_prompt = build_final_prompt(prompt, intent)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                        {"role": "user",   "content": final_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2500,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"❌ Groq API error: `{str(e)}`"

            # ── Step 2: Extract & save for self-learning ─────
            generated_query = extract_pql(reply)
            save_training_example(prompt, generated_query)

            # ── Step 3: Validate (optional toggle) ───────────
            if st.session_state.validate and generated_query.strip():
                validated    = validate_pql(generated_query)
                final_reply  = (
                    f"{reply}\n\n"
                    f"---\n\n"
                    f"**🔎 Validator Result:**\n\n{validated}"
                )
            else:
                final_reply = reply

            st.markdown(final_reply)

    st.session_state.messages.append({"role": "assistant", "content": final_reply})
