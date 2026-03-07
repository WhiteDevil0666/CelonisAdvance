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

set_background("background.png")

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant AI.

Rules:

1. Use ONLY Celonis documentation provided in context.
2. Preserve PQL syntax exactly.
3. Never convert PQL to SQL.
4. Never invent functions.
5. If documentation does not contain the answer say:
   "Not found in official Celonis documentation."
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

        st.warning("Vector store not found")

        return None, []


index, metadata = load_vector_store()

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
        "celonis","pql","pu_","datediff",
        "pull-up","throughput","event log"
    ]

    return any(word in prompt.lower() for word in keywords)

# =====================================
# DETECT PQL FUNCTION
# =====================================

def detect_pql_function(query):

    pattern = r"\b(PU_[A-Z_]+|DATEDIFF|RUNNING_SUM|ACTIVATION_COUNT)\b"

    match = re.search(pattern, query.upper())

    if match:
        return match.group(1)

    return None

# =====================================
# EXACT FUNCTION RETRIEVAL
# =====================================

def exact_function_match(query):

    function_name = detect_pql_function(query)

    if not function_name:
        return None

    for item in metadata:

        text = item.get("text","").upper()

        if function_name in text:

            return item["text"]

    return None

# =====================================
# SEMANTIC SEARCH
# =====================================

def semantic_search(query, top_k=5):

    if index is None:

        return ""

    query_embedding = EMBED_MODEL.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, top_k)

    results = []

    for idx in I[0]:

        if idx < len(metadata):

            results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================
# HYBRID RETRIEVAL
# =====================================

def retrieve_context(prompt):

    function_context = exact_function_match(prompt)

    semantic_context = semantic_search(prompt)

    if function_context:

        return f"""
FUNCTION DOCUMENTATION
----------------------
{function_context}

ADDITIONAL DOCUMENTATION
------------------------
{semantic_context}
"""

    return semantic_context

# =====================================
# HEADER
# =====================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Powered by Divyansh")

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

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    if is_celonis_query(prompt):

        context = retrieve_context(prompt)

        final_prompt = f"""
You MUST answer using ONLY the Celonis documentation below.

==============================
OFFICIAL CELOINIS DOCUMENTATION
==============================
{context}

==============================
USER QUESTION
==============================
{prompt}

Rules:

1. Preserve PQL syntax exactly.
2. Do not convert PQL to SQL.
3. Do not invent functions.
4. If documentation does not contain the answer say:

"Not found in official Celonis documentation."
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

            known_functions = [
                "PU_SUM","PU_AVG","PU_MIN","PU_MAX",
                "PU_FIRST","PU_LAST","PU_COUNT",
                "PU_COUNT_DISTINCT","PU_MEDIAN",
                "PU_STDEV"
            ]

            detected = re.findall(r'PU_[A-Z_]+', reply)

            for f in detected:

                if f not in known_functions:

                    reply += "\n\n⚠️ Warning: Possible non-standard PQL function detected."

            st.markdown(reply)

    st.session_state.messages.append({
        "role":"assistant",
        "content":reply
    })
