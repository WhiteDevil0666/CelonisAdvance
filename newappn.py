import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Celonis Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# BACKGROUND
# =====================================================

def set_background(image_file):

    with open(image_file,"rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>

    .stApp {{
        background:
        linear-gradient(rgba(0,0,0,0.85),rgba(0,0,0,0.85)),
        url("data:image/png;base64,{encoded}");
        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }}

    .stChatMessage {{
        background-color: rgba(20,20,30,0.85);
        border-radius:12px;
        padding:12px;
    }}

    div[data-testid="stChatInput"] {{
        background-color: rgba(20,20,30,0.95);
        border-radius:12px;
        padding:8px;
    }}

    h1,h2,h3,p,div,span {{
        color:white !important;
    }}

    </style>
    """

    st.markdown(css,unsafe_allow_html=True)

set_background("background.png")

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant.

Important Rules:

1. Use ONLY the Celonis documentation provided in the context.
2. Preserve Celonis PQL syntax exactly.
3. Never convert PQL to SQL.
4. Never invent functions.
5. Celonis relationships are defined in the data model.
6. Do NOT write SQL join conditions like:
   Orders.CustomerID = Customers.CustomerID
7. Avoid SQL concepts like JOIN, WHERE, GROUP BY.

If information is not found in documentation respond:

"Not found in official Celonis documentation."
"""

# =====================================================
# LOAD VECTOR DATABASE
# =====================================================

@st.cache_resource
def load_vector_store():

    try:

        index = faiss.read_index("pql_faiss.index")

        with open("pql_metadata.pkl","rb") as f:

            metadata = pickle.load(f)

        return index,metadata

    except:

        st.warning("Vector store could not be loaded")

        return None,[]

index,metadata = load_vector_store()

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =====================================================
# DETECT CELOINIS QUERY
# =====================================================

def is_celonis_query(prompt):

    keywords = [
        "celonis","pql","pu_",
        "datediff","pull-up",
        "throughput","event log"
    ]

    return any(k in prompt.lower() for k in keywords)

# =====================================================
# DETECT PQL FUNCTION
# =====================================================

def detect_pql_function(query):

    pattern = r"\b(PU_[A-Z_]+|DATEDIFF|RUNNING_SUM|ACTIVATION_COUNT)\b"

    match = re.search(pattern,query.upper())

    if match:
        return match.group(1)

    return None

# =====================================================
# FUNCTION FIRST RETRIEVAL
# =====================================================

def retrieve_function_doc(query):

    function = detect_pql_function(query)

    if not function:
        return None

    for item in metadata:

        text = item.get("text","").upper()

        if function in text:

            return item["text"]

    return None

# =====================================================
# SEMANTIC SEARCH
# =====================================================

def semantic_search(query,top_k=5):

    if index is None:

        return ""

    embedding = EMBED_MODEL.encode([query])

    embedding = np.array(embedding).astype("float32")

    faiss.normalize_L2(embedding)

    D,I = index.search(embedding,top_k)

    results = []

    for idx in I[0]:

        if idx < len(metadata):

            results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================================
# HYBRID RETRIEVAL
# =====================================================

def retrieve_context(prompt):

    function_doc = retrieve_function_doc(prompt)

    semantic_doc = semantic_search(prompt)

    if function_doc:

        return f"""
FUNCTION DOCUMENTATION
----------------------
{function_doc}

RELATED DOCUMENTATION
----------------------
{semantic_doc}
"""

    return semantic_doc

# =====================================================
# HEADER
# =====================================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Powered by Divyansh")

# =====================================================
# CHAT HISTORY
# =====================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =====================================================
# CHAT INPUT
# =====================================================

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    if is_celonis_query(prompt):

        context = retrieve_context(prompt)

        final_prompt = f"""
You MUST answer using ONLY the documentation below.

=============================
OFFICIAL CELOINIS DOCUMENTATION
=============================
{context}

=============================
USER QUESTION
=============================
{prompt}

Rules:

• Preserve Celonis syntax exactly
• Never use SQL
• Never write join conditions
• Only use documented PQL functions

If documentation does not contain the answer respond exactly with:

Not found in official Celonis documentation.
"""

    else:

        final_prompt = prompt

    with st.chat_message("assistant"):

        with st.spinner("Analyzing documentation..."):

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    *st.session_state.messages[:-1],
                    {"role":"user","content":final_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            reply = response.choices[0].message.content

            # =====================================
            # GUARDRAIL AGAINST FAKE FUNCTIONS
            # =====================================

            valid_functions = [
                "PU_SUM","PU_AVG","PU_MIN","PU_MAX",
                "PU_FIRST","PU_LAST","PU_COUNT",
                "PU_COUNT_DISTINCT","PU_MEDIAN",
                "PU_STDEV"
            ]

            found = re.findall(r'PU_[A-Z_]+',reply)

            for f in found:

                if f not in valid_functions:

                    reply += "\n\n⚠️ Warning: Possible non-standard PQL function detected."

            # =====================================
            # REMOVE SQL JOIN STYLE
            # =====================================

            if "=" in reply and "CUSTOMER" in reply.upper():

                reply += "\n\n⚠️ Celonis relationships are defined in the data model. SQL-style joins are not required."

            st.markdown(reply)

    st.session_state.messages.append({
        "role":"assistant",
        "content":reply
    })
