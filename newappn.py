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
# BACKGROUND + FULL DARK CSS
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
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }}
    html, body {{
        background-color: #0d0d1a !important;
        color: #e2e2f0 !important;
    }}

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

    /* ── Code blocks (dark forced) ── */
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

    /* ── Chat input (nuclear white fix) ── */
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

# ════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant specialised in PQL.

Never generate SQL. Always follow Celonis PQL syntax.

Column format: "TABLE"."COLUMN"

Pull-up syntax:
PU_FUNCTION(
    DOMAIN_TABLE("TABLE"),
    "TABLE"."COLUMN"
)

Rules:
- Use FILTER inside PU_ functions when filtering is needed
- Use DATEDIFF for date calculations
- Use ACTIVATION_COUNT for rework detection
- Use SOURCE / TARGET for process flow filtering
- Use CASE WHEN for conditional logic

Output format:

**📌 PQL Query:**
```
<query here>
```

**📖 Explanation:**
Explain step by step.

**💡 Note:**
Tips or what to replace.
"""

# ════════════════════════════════════════════════════════════
# VALIDATOR PROMPT
# ════════════════════════════════════════════════════════════

VALIDATOR_PROMPT = """
You are a strict Celonis PQL syntax validator.

Check for:
  • Missing or extra commas
  • Wrong FILTER syntax
  • Invalid DOMAIN_TABLE usage
  • SQL keywords (SELECT, FROM, WHERE, GROUP BY, JOIN)
  • Wrong column format (must be "TABLE"."COLUMN")
  • Invalid or invented function names

Respond ONLY in this format:

**✅ Validation Status:** <PASSED or ISSUES FOUND>

**🔧 Corrected PQL Query:**
```
<corrected query or original if clean>
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

if "validate" not in st.session_state:
    st.session_state.validate = True

# ════════════════════════════════════════════════════════════
# SAFE METADATA NORMALISER
# ════════════════════════════════════════════════════════════

def normalize_metadata(item):
    if isinstance(item, dict):
        return item
    return {
        "function": "PQL",
        "question": "Example",
        "answer":   str(item)
    }

# ════════════════════════════════════════════════════════════
# EXACT FUNCTION MATCH
# ════════════════════════════════════════════════════════════

def exact_function_match(query):
    if not vector_metadata:
        return []

    q       = query.upper()
    results = []
    seen    = set()

    for raw in vector_metadata:
        item = normalize_metadata(raw)
        func = item.get("function", "").upper()
        if func and func in q and func not in seen:
            results.append(item)
            seen.add(func)

    return results

# ════════════════════════════════════════════════════════════
# SEMANTIC SEARCH
# ════════════════════════════════════════════════════════════

def semantic_search(query, top_k=6):
    if vector_index is None or not vector_metadata:
        return []

    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb, dtype="float32")
    faiss.normalize_L2(emb)

    _, I = vector_index.search(emb, top_k)

    results = []
    for idx in I[0]:
        if idx < len(vector_metadata):
            results.append(normalize_metadata(vector_metadata[idx]))

    return results

# ════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ════════════════════════════════════════════════════════════

def retrieve_context(prompt):
    exact    = exact_function_match(prompt)
    semantic = semantic_search(prompt)

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
# PATTERN ENGINE
# ════════════════════════════════════════════════════════════

PATTERN_TEMPLATES = {
    "throughput": {
        "label": "⏱ Throughput / Duration",
        "query": (
            "DATEDIFF(\n"
            "    'day',\n"
            "    PU_MIN(\n"
            "        DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "        \"<EVENT_TABLE>\".\"TIMESTAMP\"\n"
            "    ),\n"
            "    PU_MAX(\n"
            "        DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "        \"<EVENT_TABLE>\".\"TIMESTAMP\"\n"
            "    )\n"
            ")"
        ),
    },
    "rework": {
        "label": "🔁 Rework Detection",
        "query": (
            "ACTIVATION_COUNT(\n"
            "    \"<EVENT_TABLE>\".\"ACTIVITY\"\n"
            ") > 1"
        ),
    },
    "first": {
        "label": "🥇 First Activity",
        "query": (
            "PU_FIRST(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<EVENT_TABLE>\".\"ACTIVITY\",\n"
            "    ORDER BY \"<EVENT_TABLE>\".\"TIMESTAMP\" ASC\n"
            ")"
        ),
    },
    "last": {
        "label": "🏁 Last Activity",
        "query": (
            "PU_LAST(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<EVENT_TABLE>\".\"ACTIVITY\",\n"
            "    ORDER BY \"<EVENT_TABLE>\".\"TIMESTAMP\" ASC\n"
            ")"
        ),
    },
    "avg": {
        "label": "📊 Average per Case",
        "query": (
            "PU_AVG(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<CASE_TABLE>\".\"<VALUE>\"\n"
            ")"
        ),
    },
    "sum": {
        "label": "➕ Sum per Case",
        "query": (
            "PU_SUM(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<CASE_TABLE>\".\"<VALUE>\"\n"
            ")"
        ),
    },
    "count": {
        "label": "🔢 Count Events",
        "query": (
            "PU_COUNT(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<EVENT_TABLE>\".\"<EVENT_ID>\"\n"
            ")"
        ),
    },
    "case_when": {
        "label": "🏷 Classify with CASE WHEN",
        "query": (
            "CASE\n"
            "    WHEN DATEDIFF('day', \"<CASE_TABLE>\".\"START\", \"<CASE_TABLE>\".\"END\") <= 5\n"
            "        THEN 'Fast'\n"
            "    WHEN DATEDIFF('day', \"<CASE_TABLE>\".\"START\", \"<CASE_TABLE>\".\"END\") <= 15\n"
            "        THEN 'Medium'\n"
            "    ELSE 'Slow'\n"
            "END"
        ),
    },
    "running_sum": {
        "label": "📈 Running Sum",
        "query": (
            "RUNNING_SUM(\n"
            "    \"<EVENT_TABLE>\".\"<VALUE>\"\n"
            ")"
        ),
    },
    "filter": {
        "label": "🔍 Filtered Aggregation",
        "query": (
            "PU_COUNT(\n"
            "    DOMAIN_TABLE(\"<CASE_TABLE>\"),\n"
            "    \"<EVENT_TABLE>\".\"<EVENT_ID>\",\n"
            "    FILTER \"<EVENT_TABLE>\".\"ACTIVITY\" = '<ACTIVITY_NAME>'\n"
            ")"
        ),
    },
}


def detect_problem(prompt):
    p = prompt.lower()

    if any(w in p for w in ["throughput", "duration", "time between", "cycle time", "lead time"]):
        return "throughput"
    if any(w in p for w in ["rework", "repeated", "loop"]):
        return "rework"
    if any(w in p for w in ["running sum", "running total", "cumulative"]):
        return "running_sum"
    if any(w in p for w in ["first activity", "first event", "first step"]):
        return "first"
    if any(w in p for w in ["last activity", "last event", "current status"]):
        return "last"
    if any(w in p for w in ["classify", "classification", "case when", "categorize", "label"]):
        return "case_when"
    if any(w in p for w in ["filter by", "only for", "where activity"]):
        return "filter"
    if any(w in p for w in ["average", "avg", "mean"]):
        return "avg"
    if any(w in p for w in ["total", "sum"]):
        return "sum"
    if "count" in p:
        return "count"

    return None


def template_query(problem):
    tmpl = PATTERN_TEMPLATES.get(problem)
    if tmpl:
        return tmpl["query"]
    return None

# ════════════════════════════════════════════════════════════
# EXTRACT PQL FROM RESPONSE
# ════════════════════════════════════════════════════════════

def extract_pql(text):
    match = re.search(r"```.*?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ════════════════════════════════════════════════════════════
# VALIDATE PQL
# ════════════════════════════════════════════════════════════

def validate_pql(query):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user",   "content": f"Validate this PQL:\n\n```\n{query}\n```"}
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

def save_training_example(question, query):
    file   = "training_data.csv"
    exists = os.path.isfile(file)
    try:
        with open(file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["time", "question", "query"])
            writer.writerow([datetime.now().isoformat(), question, query])
    except Exception:
        pass

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.session_state.validate = st.toggle(
        "🔎 Auto-validate PQL",
        value=True,
        help="Runs a second LLM pass to check and fix syntax errors."
    )

    st.markdown("---")
    st.markdown("### 💬 Chat")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Pattern Templates")
    st.markdown(
        "<small style='color:#6060aa'>Instant ready-to-use PQL templates</small>",
        unsafe_allow_html=True
    )
    for pat, tmpl in PATTERN_TEMPLATES.items():
        if st.button(tmpl["label"], key=f"pat_{pat}"):
            msg = (
                f"**📌 PQL Template — {tmpl['label']}:**\n\n"
                f"```\n{tmpl['query']}\n```\n\n"
                f"💡 Replace `<CASE_TABLE>`, `<EVENT_TABLE>`, `<VALUE>` "
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
    "*Powered by Divyansh · Ask anything or say: \"Write a PQL query for...\"*"
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

if prompt := st.chat_input("Ask a Celonis PQL question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Pattern engine first ─────────────────────────────────
    problem  = detect_problem(prompt)
    pat_hint = template_query(problem) if problem else None

    if pat_hint:
        # Use pattern as base, let LLM adapt it with context
        context      = retrieve_context(prompt)
        final_prompt = (
            f"PATTERN HINT — use as base syntax and adapt to the user's requirement:\n"
            f"```\n{pat_hint}\n```\n\n"
            f"Knowledge Base Context:\n"
            f"{'━' * 40}\n{context}\n{'━' * 40}\n\n"
            f"User Requirement:\n{prompt}\n\n"
            f"Adapt the pattern above to exactly match the user's requirement. "
            f"Use their actual table/column names if given, otherwise use clear placeholders."
        )
    else:
        context      = retrieve_context(prompt)
        final_prompt = (
            f"User Question:\n{prompt}\n\n"
            f"Knowledge Base Context:\n"
            f"{'━' * 40}\n{context}\n{'━' * 40}\n\n"
            f"Write correct Celonis PQL to answer the question."
        )

    # ── Build history (exclude latest) ──────────────────────
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Step 1: Generate
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

            # Step 2: Extract & save
            generated_query = extract_pql(reply)
            save_training_example(prompt, generated_query)

            # Step 3: Validate
            if st.session_state.validate and generated_query.strip():
                validated   = validate_pql(generated_query)
                final_reply = (
                    f"{reply}\n\n"
                    f"---\n\n"
                    f"**🔎 Validator Result:**\n\n{validated}"
                )
            else:
                final_reply = reply

            st.markdown(final_reply)

    st.session_state.messages.append({"role": "assistant", "content": final_reply})
