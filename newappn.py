import streamlit as st
from groq import Groq
import faiss
import numpy as np
import pickle
import re
import base64
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Process Mining Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# BACKGROUND IMAGE (SAFE)
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}
        h1, h2, h3, p, div, span {{
            color: white !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =====================================
# SAFE MODEL LOADING
# =====================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

@st.cache_resource(show_spinner=False)
def load_vector_store():
    try:
        index = faiss.read_index("pql_faiss.index")
        with open("pql_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except:
        return None, None

EMBED_MODEL = load_embedding_model()
index, metadata = load_vector_store()

# =====================================
# SYSTEM PROMPT (UNCHANGED)
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

STRICT RULES:

1. For general Process Mining questions:
   - Provide structured explanation.
   - Use practical examples.

2. For Celonis / PQL questions:
   - STRICTLY use provided documentation context.
   - Preserve official syntax EXACTLY.
   - Do NOT convert PQL into SQL.
   - Do NOT invent examples.
   - If not found in documentation, say:
     "Not found in official Celonis documentation."

3. Only generate new PQL queries if explicitly requested.
4. Never use SQL keywords.
5. Priority: Technical Accuracy > Simplicity.
"""

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_mode" not in st.session_state:
    st.session_state.file_mode = False

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

# =====================================
# FILE UPLOAD (SAFE)
# =====================================

st.sidebar.header("📂 Upload Process File")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, Excel or TXT (Max 5MB)",
    type=["pdf", "xlsx", "xls", "txt"]
)

MAX_FILE_SIZE_MB = 5

if uploaded_file:

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.sidebar.error("File too large (Max 5MB)")
    else:
        file_text = ""

        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        file_text += page_text

            elif "sheet" in uploaded_file.type:
                df = pd.read_excel(uploaded_file)
                file_text = df.head(500).to_string()  # limit rows

            elif uploaded_file.type == "text/plain":
                file_text = uploaded_file.read().decode()

            # Limit file size to avoid LLM overload
            file_text = file_text[:8000]

            st.session_state.file_mode = True
            st.session_state.file_text = file_text
            st.sidebar.success("File loaded successfully!")

        except Exception:
            st.sidebar.error("File processing failed.")

# =====================================
# QUERY DETECTION
# =====================================

def is_celonis_query(prompt):
    keywords = ["celonis", "pql", "pu_", "datediff", "pull-up"]
    return any(word in prompt.lower() for word in keywords)

# =====================================
# EXACT FUNCTION MATCH
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
# SEMANTIC SEARCH (SAFE)
# =====================================

def semantic_search(query, top_k=5):
    if not index or not EMBED_MODEL:
        return ""

    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    return "\n\n".join([metadata[i]["text"] for i in I[0]])

# =====================================
# CONTEXT RETRIEVAL
# =====================================

def retrieve_context(prompt):
    exact = exact_function_match(prompt)
    if exact:
        return exact
    return semantic_search(prompt)

# =====================================
# SAFE LLM CALL
# =====================================

def call_llm(messages):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception:
        return "⚠️ AI service temporarily unavailable. Please try again."

# =====================================
# MAIN HEADER
# =====================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Powered by Divyansh")

# =====================================
# DISPLAY CHAT
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

    elif st.session_state.file_mode:
        final_prompt = f"""
Answer based only on uploaded file content:

{st.session_state.file_text}

User Question:
{prompt}
"""
    else:
        final_prompt = prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            reply = call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages[:-1],
                {"role": "user", "content": final_prompt}
            ])

            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
