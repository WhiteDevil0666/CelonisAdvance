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
        h1,h2,h3,p,div,span {{
            color: white !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_background("background.png")

# =====================================
# MODELS
# =====================================

MODEL_DOC = "llama-3.1-8b-instant"
MODEL_REASONING = "openai/gpt-oss-120b"
MODEL_PQL = "llama-3.3-70b-versatile"

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBED_MODEL = load_embedding_model()

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

CRITICAL PQL RULES:
- Celonis PQL is NOT SQL.
- Never use SELECT, FROM, JOIN, GROUP BY, WHERE.
- Aggregations at dimension level must use PU functions.
- If user says "for each", you must use PU_* with target table.
- Do not manually exclude dimensions unless explicitly required.
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
    except:
        return None, []

index, metadata = load_vector_store()

# =====================================
# SESSION
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# QUERY CLASSIFIER
# =====================================

def classify_query(prompt):
    lower = prompt.lower()

    if any(word in lower for word in [
        "calculate",
        "build",
        "write",
        "generate",
        "ratio",
        "sum",
        "count",
        "avg",
        "for each"
    ]):
        return "pql"

    if any(word in lower for word in [
        "why",
        "impact",
        "analysis",
        "optimize"
    ]):
        return "reasoning"

    return "documentation"

# =====================================
# VALIDATION ENGINES
# =====================================

SQL_KEYWORDS = ["SELECT", "FROM", "JOIN", "GROUP BY", "WHERE"]

def contains_sql(text):
    return any(k in text.upper() for k in SQL_KEYWORDS)

def missing_pu_target(text):
    """
    Detect if PU function is missing target table.
    """
    return re.search(r'PU_\w+\(\s*"[A-Za-z0-9_]+"\s*\)', text) is None

def illegal_dimension_filter(text):
    """
    Prevent manual dimension exclusion.
    """
    return "!=" in text and "companycode" in text.lower()

# =====================================
# STRICT PQL GENERATOR
# =====================================

def generate_pql(prompt):

    strict_prompt = f"""
Convert the following requirement into VALID Celonis PQL.

IMPORTANT:
- If requirement contains "for each", use PU functions.
- Syntax must be: PU_SUM( target_table , source_table.column , filter )
- Do NOT exclude dimension manually unless explicitly asked.
- No SQL allowed.

Requirement:
{prompt}

Return ONLY valid Celonis PQL.
"""

    response = client.chat.completions.create(
        model=MODEL_PQL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": strict_prompt}
        ],
        temperature=0.0,
        max_tokens=800
    )

    output = response.choices[0].message.content

    # HARD FIX LOOP
    if contains_sql(output) or illegal_dimension_filter(output):

        correction_prompt = f"""
Your previous answer violated Celonis rules.
Fix it.

- No SQL.
- No manual dimension exclusion.
- Use proper PU syntax with target table.

Original:
{output}
"""

        correction = client.chat.completions.create(
            model=MODEL_PQL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )

        return correction.choices[0].message.content

    return output

# =====================================
# DOCUMENTATION SEARCH
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

def semantic_search(query, top_k=5):
    if not index:
        return ""
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return "\n\n".join([metadata[i]["text"] for i in I[0]])

def retrieve_context(prompt):
    exact = exact_function_match(prompt)
    if exact:
        return exact
    return semantic_search(prompt)

# =====================================
# UI HEADER
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

    query_type = classify_query(prompt)

    # ===============================
    # 🔥 PQL GENERATION ROUTE
    # ===============================

    if query_type == "pql":

        pql_output = generate_pql(prompt)

        with st.chat_message("assistant"):
            st.markdown("### 📊 Generated Celonis PQL")
            st.code(pql_output, language="sql")

        st.session_state.messages.append({
            "role": "assistant",
            "content": pql_output
        })

    # ===============================
    # 🔍 DOCUMENTATION / REASONING
    # ===============================

    else:

        if "pql" in prompt.lower():
            context = retrieve_context(prompt)
            final_prompt = f"""
Documentation Context:
{context}

User Question:
{prompt}
"""
        else:
            final_prompt = prompt

        selected_model = MODEL_REASONING if query_type == "reasoning" else MODEL_DOC

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
