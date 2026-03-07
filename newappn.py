import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import base64
import csv
import os
from datetime import datetime


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Celonis Process Mining Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# BACKGROUND
# ==========================================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_style = f"""
        background:
        linear-gradient(rgba(0,0,0,0.83),rgba(0,0,0,0.83)),
        url("data:image/png;base64,{encoded}");
        background-size:cover;
        background-position:center;
        background-attachment:fixed;
        """
    except:
        bg_style = "background:linear-gradient(135deg,#0d0d1a,#141428,#0f1a2e);"

    css = f"""
    <style>
    .stApp {{ {bg_style} }}
    pre {{
        background:#13131f !important;
        border-radius:10px;
        padding:16px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background("background.png")


# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================
# VALID PQL FUNCTIONS
# ==========================================================

VALID_PQL_FUNCTIONS = [
    "PU_COUNT", "PU_SUM", "PU_AVG", "PU_MIN", "PU_MAX",
    "PU_FIRST", "PU_LAST", "PU_COUNT_DISTINCT",
    "DATEDIFF", "RUNNING_SUM", "ACTIVATION_COUNT",
    "SOURCE", "TARGET", "CALC_THROUGHPUT", "CALC_REWORK"
]


# ==========================================================
# PQL FUNCTION BRAIN
# ==========================================================

PQL_FUNCTION_BRAIN = {
    "throughput":       "DATEDIFF",
    "cycle time":       "DATEDIFF",
    "lead time":        "DATEDIFF",
    "duration":         "DATEDIFF",
    "average":          "PU_AVG",
    "mean":             "PU_AVG",
    "sum":              "PU_SUM",
    "total":            "PU_SUM",
    "count":            "PU_COUNT",
    "minimum":          "PU_MIN",
    "maximum":          "PU_MAX",
    "first activity":   "PU_FIRST",
    "last activity":    "PU_LAST",
    "distinct":         "PU_COUNT_DISTINCT",
    "running":          "RUNNING_SUM",
    "rework":           "ACTIVATION_COUNT"
}


def detect_best_function(prompt):
    p = prompt.lower()
    for key, func in PQL_FUNCTION_BRAIN.items():
        if key in p:
            return func
    return None


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are a Senior Celonis Process Mining Consultant specialised in PQL.

Never produce SQL.

Only produce Celonis PQL.

Column format:

"TABLE"."COLUMN"

Pull-up syntax:

PU_SUM(
 DOMAIN_TABLE("CASE_TABLE"),
 "CASE_TABLE"."VALUE"
)

Return answers like:

📌 PQL Query

QUERY


📖 Explanation
Explain step by step.

💡 Note
Tips if required.
"""


# ==========================================================
# VALIDATOR PROMPT
# ==========================================================

VALIDATOR_PROMPT = f"""
You are a strict Celonis PQL syntax validator.

Valid functions:
{",".join(VALID_PQL_FUNCTIONS)}

Check:

- SQL keywords
- wrong column format
- missing commas
- invalid DOMAIN_TABLE usage

Return:

Validation Status: PASSED or ISSUES FOUND
Corrected Query:
<query>
Notes:
<issues>
"""


# ==========================================================
# LOAD VECTOR STORE
# ==========================================================

@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index("pql_knowledge.index")
        with open("pql_knowledge.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.warning(f"⚠️ Knowledge base not found ({e}). Running in LLM-only mode.")
        return None, []


vector_index, vector_metadata = load_vector_store()


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "validate" not in st.session_state:
    st.session_state.validate = True


# ==========================================================
# NORMALIZE METADATA
# ==========================================================

def normalize_metadata(item):
    if isinstance(item, dict):
        return item
    return {
        "function": "PQL",
        "question": "Example",
        "answer":   str(item)
    }


# ==========================================================
# EXACT MATCH SEARCH
# ==========================================================

def exact_function_match(query):
    if not vector_metadata:
        return []

    q       = query.upper()
    results = []
    seen    = set()

    for raw in vector_metadata:
        item = normalize_metadata(raw)
        func = item.get("function", "").upper()
        if func and func in q and func not in seen:
            results.append(item)
            seen.add(func)

    return results


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

def semantic_search(query, top_k=6):
    if vector_index is None:
        return []

    emb = EMBED_MODEL.encode([query])
    emb = np.array(emb).astype("float32")
    faiss.normalize_L2(emb)

    _, I = vector_index.search(emb, top_k)

    results = []
    for idx in I[0]:
        if idx < len(vector_metadata):
            results.append(
                normalize_metadata(vector_metadata[idx])
            )

    return results


# ==========================================================
# CONTEXT BUILDER
# ==========================================================

def retrieve_context(prompt):
    exact    = exact_function_match(prompt)
    semantic = semantic_search(prompt)
    combined = exact + semantic

    if not combined:
        return ""

    text = ""
    for item in combined[:6]:
        text += f"""
Function: {item.get("function")}

Example Question:
{item.get("question")}

PQL Answer:
{item.get("answer")}

---
"""
    return text


# ==========================================================
# PATTERN ENGINE
# ==========================================================

PATTERN_TEMPLATES = {

    "throughput": """
DATEDIFF(
 'day',
 PU_MIN(
  DOMAIN_TABLE("<CASE_TABLE>"),
  "<EVENT_TABLE>"."TIMESTAMP"
 ),
 PU_MAX(
  DOMAIN_TABLE("<CASE_TABLE>"),
  "<EVENT_TABLE>"."TIMESTAMP"
 )
)
""",

    "rework": """
ACTIVATION_COUNT(
 "<EVENT_TABLE>"."ACTIVITY"
) > 1
""",

    "first": """
PU_FIRST(
 DOMAIN_TABLE("<CASE_TABLE>"),
 "<EVENT_TABLE>"."ACTIVITY"
)
""",

    "last": """
PU_LAST(
 DOMAIN_TABLE("<CASE_TABLE>"),
 "<EVENT_TABLE>"."ACTIVITY"
)
""",

    "avg": """
PU_AVG(
 DOMAIN_TABLE("<CASE_TABLE>"),
 "<CASE_TABLE>"."<VALUE>"
)
""",

    "sum": """
PU_SUM(
 DOMAIN_TABLE("<CASE_TABLE>"),
 "<CASE_TABLE>"."<VALUE>"
)
""",

    "count": """
PU_COUNT(
 DOMAIN_TABLE("<CASE_TABLE>"),
 "<EVENT_TABLE>"."<EVENT_ID>"
)
"""
}


def detect_problem(prompt):
    p = prompt.lower()

    if "throughput" in p or "duration" in p:
        return "throughput"
    if "rework" in p:
        return "rework"
    if "first activity" in p:
        return "first"
    if "last activity" in p:
        return "last"
    if "average" in p:
        return "avg"
    if "sum" in p:
        return "sum"
    if "count" in p:
        return "count"

    return None


# ==========================================================
# ACTIVITY EXTRACTION
# ==========================================================

def extract_activities(prompt):
    pattern = r"between (.*?) and (.*)"
    match   = re.search(pattern, prompt.lower())

    if match:
        start = match.group(1).strip().title()
        end   = match.group(2).strip().title()
        return start, end

    return None, None


# ==========================================================
# BUILD THROUGHPUT QUERY
# ==========================================================

def build_activity_throughput_query(start_activity, end_activity):
    query = f"""
DATEDIFF(
 'day',
 PU_FIRST(
  DOMAIN_TABLE("<CASE_TABLE>"),
  "<EVENT_TABLE>"."TIMESTAMP",
  FILTER "<EVENT_TABLE>"."ACTIVITY" = '{start_activity}'
 ),
 PU_FIRST(
  DOMAIN_TABLE("<CASE_TABLE>"),
  "<EVENT_TABLE>"."TIMESTAMP",
  FILTER "<EVENT_TABLE>"."ACTIVITY" = '{end_activity}'
 )
)
"""
    return query


# ==========================================================
# EXTRACT PQL
# ==========================================================

def extract_pql(text):
    match = re.search(r"```(?:pql)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ==========================================================
# VALIDATE PQL
# ==========================================================

def validate_pql(query):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user",   "content": query}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Validator error: {e}"


# ==========================================================
# SAVE TRAINING DATA
# ==========================================================

def save_training_example(question, query):
    file   = "training_data.csv"
    exists = os.path.isfile(file)

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time", "question", "query"])
        writer.writerow([datetime.now(), question, query])


# ==========================================================
# AUTO TRAIN VECTOR STORE
# ==========================================================

def auto_train_vector_store(question, query):
    if vector_index is None:
        return

    text = f"{question}\n{query}"
    emb  = EMBED_MODEL.encode([text])
    emb  = np.array(emb).astype("float32")

    try:
        vector_index.add(emb)
        vector_metadata.append({
            "function": "AUTO_LEARNED",
            "question": question,
            "answer":   query
        })
    except:
        pass


# ==========================================================
# UI
# ==========================================================

st.title("🧠 Celonis Process Mining Copilot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================================================
# CHAT INPUT
# ==========================================================

if prompt := st.chat_input("Ask Celonis PQL question"):

    st.session_state.messages.append({
        "role":    "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    problem                      = detect_problem(prompt)
    start_activity, end_activity = extract_activities(prompt)

    if problem == "throughput" and start_activity and end_activity:
        query = build_activity_throughput_query(start_activity, end_activity)
        reply = f"""
📌 PQL Query

{query}


Throughput between **{start_activity}** and **{end_activity}**
"""

    elif problem:
        query = PATTERN_TEMPLATES[problem]
        reply = f"""
📌 PQL Query

{query}


Generated using template.
"""

    else:
        context      = retrieve_context(prompt)
        final_prompt = f"""
User Question:
{prompt}

Context:
{context}

Write correct Celonis PQL.
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": final_prompt}
            ],
            temperature=0.1
        )
        reply = response.choices[0].message.content

    query      = extract_pql(reply)
    validation = validate_pql(query)

    save_training_example(prompt, query)
    auto_train_vector_store(prompt, query)

    final = f"""
{reply}

---

🔎 Validator Result

{validation}
"""

    with st.chat_message("assistant"):
        st.markdown(final)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": final
    })
