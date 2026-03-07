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
# FULL CSS — ALL WHITE AREAS FIXED
# =====================================

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

    /* ─── Base app ─── */
    .stApp {{
        {bg_style}
    }}
    .main, section.main > div, .block-container {{
        background-color: transparent !important;
    }}
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }}

    /* ─── KILL ALL WHITE BACKGROUNDS globally ─── */
    *, *::before, *::after {{
        box-sizing: border-box;
    }}

    /* Default text */
    html, body {{
        background-color: #0d0d1a !important;
        color: #e2e2f0 !important;
    }}

    /* Every div/span/p text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown span, .stMarkdown div,
    p, span, label, li, td, th {{
        color: #e2e2f0 !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: #ffffff !important;
    }}

    strong, b {{
        color: #a78bfa !important;
    }}

    em, i {{
        color: #94a3b8 !important;
    }}

    /* ─── Chat messages ─── */
    .stChatMessage {{
        background-color: rgba(18, 18, 32, 0.92) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 8px !important;
    }}

    /* ─── CODE BLOCKS — DARK FORCED ─── */
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
    .stCodeBlock {{
        background-color: #13131f !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 10px !important;
    }}
    .stCodeBlock code,
    div[data-testid="stCode"] code {{
        background-color: transparent !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    div[data-testid="stCode"] {{
        background-color: #13131f !important;
    }}

    /* ─── CHAT INPUT — NUCLEAR WHITE BG FIX ─── */
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
    /* The actual textarea element */
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
        -webkit-box-shadow: none !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #6060aa !important;
        opacity: 1 !important;
    }}
    /* Send button */
    div[data-testid="stChatInput"] button,
    div[data-testid="stChatInput"] button:hover {{
        background-color: rgba(139,92,246,0.35) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }}

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {{
        background-color: #0b0b18 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: #e2e2f0 !important;
    }}
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
        transition: all 0.2s ease;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background-color: rgba(139,92,246,0.25) !important;
        border-color: rgba(139,92,246,0.6) !important;
    }}

    /* ─── Selectbox ─── */
    div[data-testid="stSelectbox"] > div > div {{
        background-color: #1a1a2e !important;
        color: #e2e2f0 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stSelectbox"] svg {{
        fill: #a78bfa !important;
    }}

    /* Dropdown options */
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: #1a1a2e !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] li {{
        color: #e2e2f0 !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{
        background-color: rgba(139,92,246,0.2) !important;
    }}

    /* ─── Horizontal divider ─── */
    hr {{
        border-color: rgba(255,255,255,0.08) !important;
    }}

    /* ─── Warning/info boxes ─── */
    div[data-testid="stAlert"] {{
        background-color: rgba(30,30,60,0.8) !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        color: #e2e2f0 !important;
        border-radius: 8px !important;
    }}

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: #0d0d1a; }}
    ::-webkit-scrollbar-thumb {{ background: #3a3a6a; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #6060aa; }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT — v3 UPGRADED
# =====================================

SYSTEM_PROMPT = """
You are an expert Celonis Process Mining Consultant AI specialising in PQL (Process Query Language).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. WRITE CUSTOM PQL QUERIES based on what the user describes.
   - Extract actual table names, column names, and conditions from the user's question.
   - Do NOT use placeholder names like ORDER_TABLE or ORDER_VALUE unless the user said them.
   - If the user gives you real table/column names, use those EXACTLY.
   - If the user does NOT provide table/column names, ask them first OR write the query
     with clearly labelled placeholders like <YOUR_TABLE>, <YOUR_COLUMN> and tell them to replace.

2. EXPLAIN PQL functions with syntax and real examples.

3. ANSWER process mining conceptual questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PQL SYNTAX RULES — STRICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Column reference:       "TABLE_NAME"."COLUMN_NAME"
✅ Pull-Up syntax:         PU_FUNCTION ( DOMAIN_TABLE ( "TABLE" ), "TABLE"."COLUMN", FILTER "TABLE"."COL" = 'value' )
✅ Date difference:        DATEDIFF ( 'day', "TABLE"."START_DATE", "TABLE"."END_DATE" )
✅ Case expression:        CASE WHEN condition THEN value ELSE other END
✅ Source/Target:          SOURCE ( "TABLE"."ACTIVITY_COL", EXCLUDE DUPLICATE ACTIVITIES ) = 'ActivityName'
✅ Running aggregation:    RUNNING_SUM ( "TABLE"."COLUMN" )
✅ Rework detection:       ACTIVATION_COUNT ( "TABLE"."ACTIVITY_COL" ) > 1

❌ NEVER use: SELECT, FROM, WHERE, GROUP BY, JOIN, AS, COUNT(*), SUM()
❌ NEVER write SQL
❌ NEVER invent function names
❌ NEVER use generic placeholders without telling the user to replace them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — ALWAYS USE THIS FOR QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**📌 PQL Query:**
```
<PQL code here — properly indented>
```

**📖 What this does:**
<plain English explanation, line by line>

**💡 Note:**
<tips, edge cases, or what to replace if placeholders used>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY: Accuracy > Context-Awareness > Simplicity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        st.warning(f"⚠️ Vector store not loaded ({e}). Running in LLM-only mode.")
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
    p = prompt.lower()
    write_kw = [
        "write", "generate", "create", "build", "give me", "make a",
        "query for", "pql for", "calculate", "compute", "show query",
        "can you write", "can you create", "help me write", "i need a query",
        "i want a query", "example query", "how would i"
    ]
    pql_kw = [
        "celonis", "pql", "pu_", "datediff", "pull-up", "throughput",
        "event log", "case duration", "activity", "variant", "kpi",
        "filter", "domain_table", "rework", "bottleneck", "cycle time",
        "lead time", "source", "target", "running_sum", "activation_count",
        "process mining", "process query"
    ]
    is_write = any(w in p for w in write_kw)
    is_pql   = any(w in p for w in pql_kw)

    if is_write and is_pql:
        return "write_pql"
    elif is_pql:
        return "explain_pql"
    else:
        return "general"

# =====================================
# VECTOR STORE SEARCH
# =====================================

def exact_function_match(query):
    if not metadata:
        return None
    tokens = re.findall(r'\bPU_[A-Z_]+\b', query.upper())
    for token in tokens:
        for item in metadata:
            if token.lower() in item.get("url", "").lower():
                return item["text"]
    return None

def semantic_search(query, top_k=8):
    if index is None or not metadata:
        return ""
    emb = EMBED_MODEL.encode([query])
    _, I = index.search(np.array(emb), top_k)
    return "\n\n---\n\n".join(
        metadata[i]["text"] for i in I[0] if i < len(metadata)
    )

def retrieve_context(prompt):
    exact = exact_function_match(prompt)
    return f"[EXACT MATCH]\n\n{exact}" if exact else semantic_search(prompt)

# =====================================
# BUILD FINAL PROMPT
# =====================================

def build_final_prompt(prompt, intent):
    if intent == "general":
        return prompt

    context = retrieve_context(prompt)

    base = f"User Requirement:\n{prompt}\n"

    if context.strip():
        ctx_block = f"""
Documentation Context (follow syntax strictly):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    else:
        ctx_block = "No documentation found. Use your expert PQL knowledge.\n\n"

    if intent == "write_pql":
        return f"""TASK: Write a custom PQL query for the user's exact requirement.

{ctx_block}{base}
IMPORTANT:
- Use the actual table/column names the user mentioned.
- If they did not mention names, use clearly labelled placeholders like <CASE_TABLE>, <ACTIVITY_COLUMN>.
- Write proper indented PQL. Do NOT write SQL.
- Explain every line after the query.
"""
    elif intent == "explain_pql":
        return f"""TASK: Explain the PQL concept the user is asking about.

{ctx_block}{base}"""

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
    st.markdown(f"**Index:** {'✅ Loaded' if index is not None else '⚠️ LLM-only'}")
    st.markdown(f"**Model:** `{MODEL_NAME}`")

# =====================================
# HEADER
# =====================================

st.title("🧠 Celonis Process Mining Copilot")
st.markdown("*Powered by Divyansh · Ask anything — or say: \"Write a PQL query for...\"*")
st.markdown("---")

# =====================================
# CHAT HISTORY
# =====================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask a question or say: 'Write a PQL query for...'"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine intent
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
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2500
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"❌ Groq API error: `{str(e)}`"

            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
