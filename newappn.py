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
        background-color: rgba(20,20,30,0.9);
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
You are a Senior Celonis Process Mining Consultant AI.

Rules:

1. Use ONLY Celonis PQL.
2. Never generate SQL.
3. Celonis relationships exist in the data model.
4. Never write SQL joins.
5. Column syntax must be:

"TABLE"."COLUMN"

6. Always show correct PQL syntax when explaining functions.
7. If the user asks about a function:
   - Explain it clearly
   - Show correct syntax
   - Provide a real example query.

If information is not found in documentation say:

"Not found in official Celonis documentation."

Always structure answers as:

📌 PQL Query

<query>

📖 Explanation

<explanation>

💡 Example

<example query>
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

        st.warning("Vector database not found")

        return None,[]

index,metadata = load_vector_store()

# =====================================================
# SESSION MEMORY
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =====================================================
# FUNCTION DETECTION
# =====================================================

def detect_pql_function(prompt):

    pattern = r"\b(PU_[A-Z_]+|DATEDIFF|RUNNING_SUM|ACTIVATION_COUNT|DOMAIN_TABLE|BIND)\b"

    match = re.search(pattern,prompt.upper())

    if match:

        return match.group(1)

    return None

# =====================================================
# FUNCTION DOCUMENTATION RETRIEVAL
# =====================================================

def retrieve_function_doc(prompt):

    function = detect_pql_function(prompt)

    if not function:

        return None

    for item in metadata:

        if function in item["text"].upper():

            return item["text"]

    return None

# =====================================================
# SEMANTIC SEARCH
# =====================================================

def semantic_search(prompt,top_k=5):

    if index is None:

        return ""

    embedding = EMBED_MODEL.encode([prompt])

    embedding = np.array(embedding).astype("float32")

    faiss.normalize_L2(embedding)

    D,I = index.search(embedding,top_k)

    results = []

    for idx in I[0]:

        if idx < len(metadata):

            results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================================
# CONTEXT PIPELINE
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
# PQL TEMPLATE ENGINE
# =====================================================

def detect_problem(prompt):

    p = prompt.lower()

    if "throughput" in p or "cycle time" in p:

        return "throughput"

    if "rework" in p:

        return "rework"

    if "variant" in p:

        return "variant"

    return None

def extract_activities(prompt):

    pattern = r"between (.+?) and (.+)"

    match = re.search(pattern,prompt.lower())

    if match:

        start = match.group(1).strip()

        end = match.group(2).strip()

        return start,end

    return None,None

def generate_template(prompt):

    problem = detect_problem(prompt)

    if problem == "throughput":

        start,end = extract_activities(prompt)

        if start and end:

            return f"""
DATEDIFF(
 'day',
 PU_FIRST(
  "CASE_TABLE",
  "ACTIVITY_TABLE"."TIMESTAMP",
  FILTER "ACTIVITY_TABLE"."ACTIVITY" = '{start}'
 ),
 PU_FIRST(
  "CASE_TABLE",
  "ACTIVITY_TABLE"."TIMESTAMP",
  FILTER "ACTIVITY_TABLE"."ACTIVITY" = '{end}'
 )
)
"""

    if problem == "rework":

        return """
ACTIVATION_COUNT(
 "ACTIVITY_TABLE"."ACTIVITY"
) > 1
"""

    if problem == "variant":

        return """
VARIANT(
 "ACTIVITY_TABLE"."ACTIVITY"
)
"""

    return None

# =====================================================
# HEADER
# =====================================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.markdown("Powered by Divyansh")

# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =====================================================
# CHAT INPUT
# =====================================================

if prompt := st.chat_input("Ask your Celonis question..."):

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    template = generate_template(prompt)

    if template:

        reply = f"""
📌 PQL Query

{template}

📖 Explanation

This query was automatically generated by the Copilot PQL template engine.

💡 Example

Replace CASE_TABLE and ACTIVITY_TABLE with your actual tables.
"""

    else:

        context = retrieve_context(prompt)

        final_prompt = f"""

Use the documentation below to answer.

-----------------------
{context}
-----------------------

User question:

{prompt}
"""

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":final_prompt}
            ],

            temperature=0.1,
            max_tokens=1500
        )

        reply = response.choices[0].message.content

    with st.chat_message("assistant"):

        st.markdown(reply)

    st.session_state.messages.append({
        "role":"assistant",
        "content":reply
    })
