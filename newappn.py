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

        /* Full App Background */
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Remove white container */
        .main {{
            background-color: transparent !important;
        }}

        section.main > div {{
            background-color: transparent !important;
        }}

        /* Remove bottom white space */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}

        /* Chat bubbles */
        .stChatMessage {{
            background-color: rgba(20, 20, 30, 0.85);
            border-radius: 12px;
            padding: 12px;
        }}

        /* Chat input */
        div[data-testid="stChatInput"] {{
            background-color: rgba(20, 20, 30, 0.95);
            border-radius: 12px;
            padding: 8px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 15, 25, 0.95);
        }}

        /* Text color */
        h1, h2, h3, p, div, span {{
            color: white !important;
        }}

        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Background image not found, skip gracefully

# Call background
set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

STRICT RULES:

1. For general Process Mining questions:
   - Provide structured explanation.
   - Use simple real-world examples.

2. For Celonis / PQL questions:
   - STRICTLY use provided documentation context AND example queries.
   - Preserve official syntax EXACTLY.
   - Do NOT convert PQL into SQL.
   - Do NOT simplify syntax.
   - Do NOT invent examples outside of provided context.
   - If not found in documentation, say:
     "Not found in official Celonis documentation."

3. Only generate new PQL queries if explicitly requested.
4. Never use SQL keywords like SELECT, FROM, WHERE, JOIN.
5. When example PQL queries are provided in context, reference them to guide your answer.
6. Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# LOAD BOTH VECTOR STORES
# =====================================

@st.cache_resource
def load_vector_stores():
    # --- Primary: Official Celonis Docs ---
    docs_index = faiss.read_index("pql_faiss.index")
    with open("pql_metadata.pkl", "rb") as f:
        docs_metadata = pickle.load(f)  # list of dicts: {url, title, text}

    # --- Secondary: PQL Q&A Examples ---
    qa_index = faiss.read_index("pql_knowledge.index")
    with open("pql_knowledge.pkl", "rb") as f:
        qa_metadata = pickle.load(f)  # list of strings: Question + PQL + Explanation

    return docs_index, docs_metadata, qa_index, qa_metadata

docs_index, docs_metadata, qa_index, qa_metadata = load_vector_stores()

# =====================================
# SESSION MEMORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# QUERY DETECTION
# =====================================

def is_celonis_query(prompt):
    keywords = [
        "celonis", "pql", "pu_", "datediff", "remap",
        "process query language", "pull-up", "running_sum",
        "throughput", "event log", "case table", "activity table",
        "conformance", "variant", "filter", "process mining",
        "count_table", "avg", "median", "moving_avg", "process",
        "avg_process", "source", "target", "rework", "loop"
    ]
    return any(word in prompt.lower() for word in keywords)

# =====================================
# EXACT FUNCTION ROUTING
# =====================================

def exact_function_match(query):
    query_upper = query.upper()

    # Match any ALL_CAPS_WITH_UNDERSCORES token (PQL function pattern)
    tokens = re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', query_upper)

    for token in tokens:
        for item in docs_metadata:
            url = item.get("url", "").lower()
            if token.lower().replace("_", "-") in url or token.lower() in url:
                return item["text"]

    return None

# =====================================
# SEMANTIC SEARCH — BOTH STORES
# =====================================

def semantic_search(query, top_k=3):
    query_embedding = EMBED_MODEL.encode([query])
    query_np = np.array(query_embedding)

    # Search official docs
    _, I_docs = docs_index.search(query_np, top_k)
    doc_results = "\n\n".join(
        docs_metadata[i]["text"] for i in I_docs[0] if i < len(docs_metadata)
    )

    # Search PQL Q&A examples
    _, I_qa = qa_index.search(query_np, top_k)
    qa_results = "\n\n".join(
        qa_metadata[i] for i in I_qa[0] if i < len(qa_metadata)
    )

    return doc_results, qa_results

# =====================================
# CONTEXT PIPELINE
# =====================================

def retrieve_context(prompt):
    # 1. Try exact function match from docs first
    exact_match = exact_function_match(prompt)

    # 2. Always run semantic search on both stores
    doc_context, qa_context = semantic_search(prompt, top_k=3)

    # 3. Build final context block
    sections = []

    if exact_match:
        sections.append("### 📖 Official Documentation (Exact Match):\n" + exact_match)
    elif doc_context:
        sections.append("### 📖 Official Documentation:\n" + doc_context)

    if qa_context:
        sections.append("### 💡 Relevant PQL Query Examples:\n" + qa_context)

    return "\n\n".join(sections)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## 🧠 Celonis Copilot")
    st.markdown("---")
    st.markdown("**Model:** `llama-3.1-8b-instant`")
    st.markdown("**Embed Model:** `all-MiniLM-L6-v2`")
    st.markdown("---")
    st.markdown("**Knowledge Base:**")
    st.markdown(f"- 📚 Docs chunks: `{len(docs_metadata)}`")
    st.markdown(f"- 💡 PQL examples: `{len(qa_metadata)}`")
    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Divyansh")

# =====================================
# UI HEADER
# =====================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Ask anything about **PQL**, **Process Mining**, or **Celonis** platform.")

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT
# =====================================

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build final prompt
    if is_celonis_query(prompt):
        context = retrieve_context(prompt)
        final_prompt = f"""STRICT MODE ENABLED.

Documentation & Example Context:
---------------------------------
{context}
---------------------------------

User Question:
{prompt}
"""
    else:
        final_prompt = prompt

    # Build API message history (use original messages for history, injected prompt for current)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages[:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": final_prompt})

    # Generate response
    reply = ""
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.1,
                    max_tokens=1500
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
            except Exception as e:
                reply = f"⚠️ Error contacting Groq API: {str(e)}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
