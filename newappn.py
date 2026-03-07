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
# BACKGROUND
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        css = f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main, section.main > div, .block-container {{
            background-color: transparent !important;
        }}

        h1,h2,h3,h4,p,span,div {{
            color:white !important;
        }}

        .stChatMessage {{
            background-color: rgba(20,20,30,0.9);
            border-radius: 12px;
            padding: 14px;
        }}

        div[data-testid="stChatInput"] {{
            background-color:#141428;
            border-radius:10px;
        }}

        pre {{
            background:#13131f !important;
            border-radius:10px;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    except:
        pass

set_background("background.png")

# =====================================
# CONFIG
# =====================================

MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are an expert Celonis Process Mining Consultant AI specialising in PQL.

CAPABILITIES

1. Write PQL queries
2. Explain PQL functions
3. Answer process mining questions
4. Debug broken PQL queries

STRICT RULES

• NEVER write SQL
• NEVER convert PQL to SQL
• Use official PQL syntax

COLUMN FORMAT
"TABLE"."COLUMN"

PULL UP SYNTAX
PU_FUNCTION ( DOMAIN_TABLE ( "TABLE" ), "TABLE"."COLUMN" )

DATE DIFF
DATEDIFF ( 'day', "TABLE"."START", "TABLE"."END" )

CASE EXPRESSION
CASE WHEN condition THEN value ELSE other END

RESPONSE FORMAT FOR QUERIES

📌 PQL Query
<code>

📖 Explanation
<step explanation>

💡 Notes
<tips>
"""

# =====================================
# LOAD VECTOR STORE
# =====================================

@st.cache_resource
def load_vector_store():

    try:

        index = faiss.read_index("pql_faiss.index")

        with open("pql_metadata.pkl","rb") as f:
            metadata = pickle.load(f)

        return index, metadata

    except:
        st.warning("Vector store not found — running LLM only mode")
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

    debug_kw = [
        "fix this query",
        "debug this query",
        "correct this pql",
        "this query gives error",
        "query not working"
    ]

    write_kw = [
        "write",
        "generate",
        "create",
        "build",
        "pql for",
        "query for",
        "calculate",
        "compute"
    ]

    pql_kw = [
        "pql",
        "celonis",
        "pu_",
        "datediff",
        "process mining",
        "event log",
        "throughput",
        "variant",
        "activity",
        "kpi",
        "rework",
        "cycle time"
    ]

    if any(d in p for d in debug_kw):
        return "debug_pql"

    if any(w in p for w in write_kw) and any(k in p for k in pql_kw):
        return "write_pql"

    if any(k in p for k in pql_kw):
        return "explain_pql"

    return "general"

# =====================================
# EXACT FUNCTION MATCH
# =====================================

def exact_function_match(query):

    tokens = re.findall(r'\bPU_[A-Z_]+\b', query.upper())

    for token in tokens:

        for item in metadata:

            url = item.get("url","").lower()

            if token.lower() in url:
                return item["text"]

    return None

# =====================================
# SEMANTIC SEARCH
# =====================================

def semantic_search(query, top_k=8):

    if index is None:
        return ""

    emb = EMBED_MODEL.encode([query])

    D,I = index.search(np.array(emb), top_k)

    results = []

    for idx in I[0]:
        results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================
# CONTEXT RETRIEVAL
# =====================================

def retrieve_context(prompt):

    exact = exact_function_match(prompt)

    if exact:
        return exact

    return semantic_search(prompt)

# =====================================
# BUILD FINAL PROMPT
# =====================================

def build_prompt(prompt, intent):

    if intent == "general":
        return prompt

    context = retrieve_context(prompt)

    if intent == "write_pql":

        return f"""
Write a PQL query.

Context
{context}

User Requirement
{prompt}

Rules
• Use correct PQL syntax
• Do not write SQL
"""

    if intent == "debug_pql":

        return f"""
Debug the following PQL query.

User Query
{prompt}

Return:

Corrected PQL Query

Explain the mistake

Explain the fix
"""

    if intent == "explain_pql":

        return f"""
Explain the following Celonis / PQL concept.

Context
{context}

Question
{prompt}
"""

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.session_state.mode = st.selectbox(
        "Response Mode",
        ["Auto","Always use docs","LLM only"]
    )

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### Examples")

    examples = [
        "Write a PQL query for average case duration",
        "Explain PU_AVG",
        "Write PQL to detect rework",
        "Fix this query PU_SUM DOMAIN_TABLE"
    ]

    for e in examples:
        if st.button(e):
            st.session_state.messages.append({"role":"user","content":e})
            st.rerun()

# =====================================
# HEADER
# =====================================

st.title("🧠 Celonis Process Mining Copilot")

st.markdown("Ask anything about **PQL or Process Mining**")

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

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    intent = detect_intent(prompt)

    if st.session_state.mode == "LLM only":
        intent = "general"

    final_prompt = build_prompt(prompt, intent)

    history = [
        {"role":m["role"],"content":m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = client.chat.completions.create(

                    model=MODEL_NAME,

                    messages=[
                        {"role":"system","content":SYSTEM_PROMPT},
                        *history,
                        {"role":"user","content":final_prompt}
                    ],

                    temperature=0.1,
                    max_tokens=2000
                )

                reply = response.choices[0].message.content

            except Exception as e:

                reply = f"API Error: {e}"

            st.markdown(reply)

    st.session_state.messages.append({"role":"assistant","content":reply})
