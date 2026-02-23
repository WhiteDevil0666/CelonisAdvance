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
# CONFIGURATION – 3 MODEL ARCHITECTURE
# =====================================

MODEL_DOC = "llama-3.1-8b-instant"          # Documentation
MODEL_REASONING = "openai/gpt-oss-120b"     # Deep reasoning
MODEL_PQL_ENGINE = "llama-3.3-70b-versatile" # Strict PQL builder

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

2. For Celonis / PQL questions:
   - STRICTLY use Celonis PQL syntax.
   - There is NO SELECT.
   - There is NO FROM.
   - There is NO JOIN.
   - There is NO GROUP BY.
   - There is NO WHERE.
   - Only use PU functions and PQL expressions.
   - Never convert PQL into SQL.

3. Only generate new PQL queries if explicitly requested.
4. Never use SQL keywords.
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
# SESSION MEMORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# QUERY CLASSIFIER
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
        "for each",
        "sum",
        "count",
        "avg"
    ]):
        return "pql_generation"

    if any(word in prompt_lower for word in [
        "why",
        "impact",
        "optimize",
        "analysis",
        "root cause"
    ]):
        return "reasoning"

    return "documentation"

# =====================================
# STRICT SQL BLOCKER
# =====================================

SQL_BLOCK_LIST = ["SELECT", "FROM", "JOIN", "GROUP BY", "WHERE"]

def contains_sql(text):
    text_upper = text.upper()
    return any(keyword in text_upper for keyword in SQL_BLOCK_LIST)

# =====================================
# STRICT PQL GENERATOR
# =====================================

def generate_strict_pql(prompt):

    strict_prompt = f"""
Convert the following business requirement into valid Celonis PQL.

CRITICAL RULES:
- Celonis PQL is NOT SQL.
- No SELECT.
- No FROM.
- No GROUP BY.
- No JOIN.
- No WHERE.
- Only use PU_SUM, PU_COUNT, PU_AVG, FILTER, column expressions.

Business Requirement:
{prompt}

Return ONLY valid PQL.
"""

    response = client.chat.completions.create(
        model=MODEL_PQL_ENGINE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": strict_prompt}
        ],
        temperature=0.0,
        max_tokens=800
    )

    output = response.choices[0].message.content

    # HARD SQL BLOCK
    if contains_sql(output):
        correction_prompt = f"""
Your previous answer used SQL syntax.
Rewrite strictly in Celonis PQL.
No SQL allowed.

Original:
{output}
"""
        correction = client.chat.completions.create(
            model=MODEL_PQL_ENGINE,
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

    # 🔹 PQL GENERATION ROUTE
    if query_type == "pql_generation":

        pql_output = generate_strict_pql(prompt)

        with st.chat_message("assistant"):
            st.markdown("### 📊 Generated Celonis PQL")
            st.code(pql_output, language="sql")

        st.session_state.messages.append({
            "role": "assistant",
            "content": pql_output
        })

    else:

        # Documentation context if needed
        if "pql" in prompt.lower():
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

        # Model selection
        if query_type == "reasoning":
            selected_model = MODEL_REASONING
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
