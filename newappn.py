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
# BACKGROUND SETUP
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        page_bg = f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{
            background-color: transparent !important;
        }}

        section.main > div {{
            background-color: transparent !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}

        .stChatMessage {{
            background-color: rgba(20, 20, 30, 0.85);
            border-radius: 12px;
            padding: 12px;
        }}

        div[data-testid="stChatInput"] {{
            background-color: rgba(20, 20, 30, 0.95);
            border-radius: 12px;
            padding: 8px;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 15, 25, 0.95);
        }}

        h1, h2, h3, p, div, span {{
            color: white !important;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    except:
        pass

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

# 🔹 3 MODEL ARCHITECTURE

MODEL_DOC = "llama-3.1-8b-instant"          # Documentation / Basic Q&A
MODEL_REASONING = "openai/gpt-oss-120b"     # Deep business reasoning
MODEL_PQL_ENGINE = "llama-3.3-70b-versatile" # Custom PQL builder

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBED_MODEL = load_embedding_model()

# =====================================
# SYSTEM PROMPT (UNCHANGED)
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

STRICT RULES:

1. For general Process Mining questions:
   - Provide structured explanation.
   - Use simple real-world examples.

2. For Celonis / PQL questions:
   - STRICTLY use provided documentation context.
   - Preserve official syntax EXACTLY.
   - Do NOT convert PQL into SQL.
   - Do NOT simplify syntax.
   - Do NOT invent examples.
   - If not found in documentation, say:
     "Not found in official Celonis documentation."

3. Only generate new PQL queries if explicitly requested.
4. Never use SQL keywords.
5. Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# LOAD VECTOR STORE (UNCHANGED)
# =====================================

@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index("pql_faiss.index")
        with open("pql_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except:
        return None, []

index, metadata = load_vector_store()

# =====================================
# SESSION MEMORY (UNCHANGED)
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# QUERY DETECTION (UNCHANGED)
# =====================================

def is_celonis_query(prompt):
    keywords = [
        "celonis", "pql", "pu_", "datediff",
        "process query language", "pull-up",
        "throughput", "event log"
    ]
    return any(word in prompt.lower() for word in keywords)

# =====================================
# SMART QUERY CLASSIFIER (NEW – SAFE ADDITION)
# =====================================

def classify_query(prompt):
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in [
        "build pql",
        "write pql",
        "generate pql",
        "create query",
        "custom kpi",
        "calculate ratio",
        "group by",
        "for each",
        "working capital"
    ]):
        return "pql_generation"

    if any(word in prompt_lower for word in [
        "why",
        "optimize",
        "improve",
        "impact",
        "analysis",
        "root cause"
    ]):
        return "reasoning"

    return "documentation"

# =====================================
# EXACT FUNCTION MATCH (UNCHANGED)
# =====================================

def exact_function_match(query):
    query_upper = query.upper()
    tokens = re.findall(r'\bPU_[A-Z_]+\b', query_upper)

    for token in tokens:
        for item in metadata:
            url = item.get("url", "").lower()
            if token.lower() in url:
                return item["text"]

    return None

# =====================================
# SEMANTIC SEARCH (UNCHANGED)
# =====================================

def semantic_search(query, top_k=5):
    if not index:
        return ""

    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================
# CONTEXT PIPELINE (UNCHANGED)
# =====================================

def retrieve_context(prompt):
    exact_match = exact_function_match(prompt)
    if exact_match:
        return exact_match
    return semantic_search(prompt)

# =====================================
# UI HEADER (UNCHANGED)
# =====================================

st.title("🧠 Process Mining Copilot(Celonis)")
st.markdown("Powered by Divyansh")

# =====================================
# DISPLAY CHAT HISTORY (UNCHANGED)
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT (UPDATED ONLY WITH ROUTING)
# =====================================

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 🔹 STRICT DOCUMENTATION CONTEXT
    if is_celonis_query(prompt):
        context = retrieve_context(prompt)

        final_prompt = f"""
STRICT MODE ENABLED.

Documentation Context:
-----------------------
{context}
-----------------------

User Question:
{prompt}
"""
    else:
        final_prompt = prompt

    # 🔹 MODEL ROUTING
    query_type = classify_query(prompt)

    if query_type == "documentation":
        selected_model = MODEL_DOC
    elif query_type == "reasoning":
        selected_model = MODEL_REASONING
    elif query_type == "pql_generation":
        selected_model = MODEL_PQL_ENGINE
    else:
        selected_model = MODEL_DOC

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *st.session_state.messages[:-1],
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
