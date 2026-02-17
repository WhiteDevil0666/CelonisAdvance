import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64
import PyPDF2
import pandas as pd
from io import BytesIO

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Process Mining Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# BACKGROUND IMAGE
# =====================================

def set_background(image_file):
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
    .stChatMessage {{
        background-color: rgba(20,20,30,0.85);
        border-radius: 12px;
        padding: 12px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: rgba(15,15,25,0.95);
    }}
    h1, h2, h3, p, div, span {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

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
# LOAD VECTOR STORE
# =====================================

@st.cache_resource
def load_vector_store():
    index = faiss.read_index("pql_faiss.index")
    with open("pql_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_vector_store()

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
# FILE UPLOAD SECTION
# =====================================

st.sidebar.header("📂 Upload Process File")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, Excel or TXT",
    type=["pdf", "xlsx", "xls", "txt"]
)

if uploaded_file:
    file_text = ""

    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            file_text += page.extract_text()

    elif "sheet" in uploaded_file.type:
        df = pd.read_excel(uploaded_file)
        file_text = df.to_string()

    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode()

    st.session_state.file_mode = True
    st.session_state.file_text = file_text
    st.sidebar.success("File loaded successfully!")

# =====================================
# ADVANCED PROCESS INTELLIGENCE
# =====================================

st.sidebar.header("🧠 Advanced Process Intelligence")

gap_btn = st.sidebar.button("🔍 Detect Process Gaps")
kpi_btn = st.sidebar.button("📊 Auto Extract KPIs")
risk_btn = st.sidebar.button("⚠️ Detect Process Risks")

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

def semantic_search(query, top_k=5):
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
# ADVANCED INTELLIGENCE FUNCTIONS
# =====================================

def run_analysis(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    return response.choices[0].message.content

if st.session_state.file_mode:

    if gap_btn:
        with st.spinner("Analyzing process gaps..."):
            result = run_analysis(f"Detect process gaps:\n{st.session_state.file_text}")
            st.markdown("## 🔍 Process Gap Analysis")
            st.markdown(result)

    if kpi_btn:
        with st.spinner("Extracting KPIs..."):
            result = run_analysis(f"Extract KPIs from:\n{st.session_state.file_text}")
            st.markdown("## 📊 Auto KPI Extraction")
            st.markdown(result)

    if risk_btn:
        with st.spinner("Detecting risks..."):
            result = run_analysis(f"Detect risks from:\n{st.session_state.file_text}")
            st.markdown("## ⚠️ Process Risk Analysis")
            st.markdown(result)

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

            response = client.chat.completions.create(
                model=MODEL_NAME,
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
