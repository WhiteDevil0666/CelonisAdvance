import pickle
import re
import base64
import pandas as pd
from io import BytesIO

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Celonis Process Mining Copilot",
    page_title="Celonis Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# SAFE BACKGROUND IMAGE
# BACKGROUND SETUP
# =====================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        page_bg = f"""
        <style>
        .stApp {{
            background:
@@ -38,34 +36,42 @@ def set_background(image_file):
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 1rem;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(15,15,25,0.95);

        .main {{
            background-color: transparent !important;
        }}
        div[data-testid="stFileUploader"] {{
            background-color: rgba(30,30,45,0.9) !important;
            padding: 15px;
            border-radius: 10px;

        section.main > div {{
            background-color: transparent !important;
        }}
        div[data-testid="stFileUploader"] * {{
            color: white !important;

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 0rem;
        }}

        .stChatMessage {{
            background-color: rgba(20,20,30,0.85);
            background-color: rgba(20, 20, 30, 0.85);
            border-radius: 12px;
            padding: 12px;
        }}
        h1, h2, h3 {{
            color: white !important;

        div[data-testid="stChatInput"] {{
            background-color: rgba(20, 20, 30, 0.95);
            border-radius: 12px;
            padding: 8px;
        }}
        p {{
            color: #dddddd !important;

        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 15, 25, 0.95);
        }}

        h1, h2, h3, p, div, span {{
            color: white !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    except:
        pass

@@ -75,7 +81,12 @@ def set_background(image_file):
# CONFIGURATION
# =====================================

MODEL_NAME = "llama-3.1-8b-instant"
# 🔹 3 MODEL ARCHITECTURE

MODEL_DOC = "llama-3.1-8b-instant"          # Documentation / Basic Q&A
MODEL_REASONING = "openai/gpt-oss-120b"     # Deep business reasoning
MODEL_PQL_ENGINE = "llama-3.3-70b-versatile" # Custom PQL builder

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
@@ -85,7 +96,7 @@ def load_embedding_model():
EMBED_MODEL = load_embedding_model()

# =====================================
# SYSTEM PROMPT
# SYSTEM PROMPT (UNCHANGED)
# =====================================

SYSTEM_PROMPT = """
@@ -95,12 +106,13 @@ def load_embedding_model():

1. For general Process Mining questions:
   - Provide structured explanation.
   - Use practical examples.
   - Use simple real-world examples.

2. For Celonis / PQL questions:
   - STRICTLY use provided documentation context.
   - Preserve official syntax EXACTLY.
   - Do NOT convert PQL into SQL.
   - Do NOT simplify syntax.
   - Do NOT invent examples.
   - If not found in documentation, say:
     "Not found in official Celonis documentation."
@@ -111,7 +123,7 @@ def load_embedding_model():
"""

# =====================================
# SAFE VECTOR STORE LOAD
# LOAD VECTOR STORE (UNCHANGED)
# =====================================

@st.cache_resource
@@ -127,143 +139,61 @@ def load_vector_store():
index, metadata = load_vector_store()

# =====================================
# SESSION STATE
# SESSION MEMORY (UNCHANGED)
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

# =====================================
# SIDEBAR FILE UPLOAD
# QUERY DETECTION (UNCHANGED)
# =====================================

st.sidebar.header("📂 Upload Process File")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, Excel or TXT (Max 5MB)",
    type=["pdf", "xlsx", "xls", "txt"]
)

if uploaded_file:
    try:
        file_text = ""

        if uploaded_file.type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    file_text += text

        elif "sheet" in uploaded_file.type:
            df = pd.read_excel(uploaded_file)
            file_text = df.to_string()

        elif uploaded_file.type == "text/plain":
            file_text = uploaded_file.read().decode()

        st.session_state.file_text = file_text[:15000]
        st.sidebar.success("File loaded successfully!")

    except:
        st.sidebar.error("File processing failed.")

# =====================================
# 🔥 ADVANCED PQL INTELLIGENCE ENGINE
# =====================================

def detect_sql_syntax(text):
    sql_keywords = ["SELECT ", " FROM ", " GROUP BY ", " ORDER BY ", " WHERE "]
    return any(keyword in text.upper() for keyword in sql_keywords)

def validate_pql_syntax(pql_text):
    if not re.search(r'\bPU_|SUM\(|COUNT\(|AVG\(', pql_text):
        return False
    if detect_sql_syntax(pql_text):
        return False
    return True

def enforce_pu_usage(prompt, generated_pql):
    dimension_keywords = ["vendor", "customer", "company", "plant", "region"]
    if any(word in prompt.lower() for word in dimension_keywords):
        if "PU_" not in generated_pql:
            return False
    return True

def build_kpi_prompt(user_prompt):
    return f"""
You are building a Celonis KPI using PQL.

STEP 1: Identify tables and aggregation level.
STEP 2: Use PU functions if aggregation is at dimension level.
STEP 3: Output ONLY valid PQL.
No explanation.

Business Question:
{user_prompt}
"""

def generate_structured_pql(user_prompt):

    structured_prompt = build_kpi_prompt(user_prompt)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": structured_prompt}
        ],
        temperature=0.1,
        max_tokens=800
    )

    generated = response.choices[0].message.content

    if not validate_pql_syntax(generated) or not enforce_pu_usage(user_prompt, generated):

        correction_prompt = f"""
Fix this PQL to follow Celonis rules.
- No SQL
- Must use PU functions if aggregating at dimension level

Original:
{generated}
"""

        correction_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=0.05,
            max_tokens=800
        )

        return correction_response.choices[0].message.content

    return generated
def is_celonis_query(prompt):
    keywords = [
        "celonis", "pql", "pu_", "datediff",
        "process query language", "pull-up",
        "throughput", "event log"
    ]
    return any(word in prompt.lower() for word in keywords)

# =====================================
# QUERY DETECTION
# SMART QUERY CLASSIFIER (NEW – SAFE ADDITION)
# =====================================

def is_celonis_query(prompt):
    keywords = ["celonis", "pql", "pu_", "datediff", "pull-up"]
    return any(word in prompt.lower() for word in keywords)
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
# EXACT FUNCTION MATCH
# EXACT FUNCTION MATCH (UNCHANGED)
# =====================================

def exact_function_match(query):
    if not metadata:
        return None

    query_upper = query.upper()
    tokens = re.findall(r'\bPU_[A-Z_]+\b', query_upper)

@@ -276,7 +206,7 @@ def exact_function_match(query):
    return None

# =====================================
# SEMANTIC SEARCH
# SEMANTIC SEARCH (UNCHANGED)
# =====================================

def semantic_search(query, top_k=5):
@@ -286,31 +216,39 @@ def semantic_search(query, top_k=5):
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    return "\n\n".join([metadata[i]["text"] for i in I[0]])
    results = []
    for idx in I[0]:
        results.append(metadata[idx]["text"])

    return "\n\n".join(results)

# =====================================
# CONTEXT PIPELINE (UNCHANGED)
# =====================================

def retrieve_context(prompt):
    exact = exact_function_match(prompt)
    if exact:
        return exact
    exact_match = exact_function_match(prompt)
    if exact_match:
        return exact_match
    return semantic_search(prompt)

# =====================================
# MAIN HEADER
# UI HEADER (UNCHANGED)
# =====================================

st.title("🧠 Process Mining Copilot (Celonis)")
st.title("🧠 Process Mining Copilot(Celonis)")
st.markdown("Powered by Divyansh")

# =====================================
# DISPLAY CHAT
# DISPLAY CHAT HISTORY (UNCHANGED)
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# CHAT INPUT
# CHAT INPUT (UPDATED ONLY WITH ROUTING)
# =====================================

if prompt := st.chat_input("Ask your question..."):
@@ -320,22 +258,8 @@ def retrieve_context(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)

    # 🔥 KPI AUTO GENERATION
    if any(word in prompt.lower() for word in ["ratio", "kpi", "calculate", "working capital", "cycle", "payment"]):

        with st.chat_message("assistant"):
            with st.spinner("Building structured KPI logic..."):
                pql_output = generate_structured_pql(prompt)
                st.markdown("### 📊 Generated PQL KPI")
                st.code(pql_output, language="sql")

        st.session_state.messages.append({
            "role": "assistant",
            "content": pql_output
        })

    elif is_celonis_query(prompt) and metadata:

    # 🔹 STRICT DOCUMENTATION CONTEXT
    if is_celonis_query(prompt):
        context = retrieve_context(prompt)

        final_prompt = f"""
@@ -349,67 +273,36 @@ def retrieve_context(prompt):
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
                    max_tokens=1200
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif st.session_state.file_text:

        final_prompt = f"""
Answer strictly using the uploaded file.

{st.session_state.file_text}

User Question:
{prompt}
"""

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1200
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
    # 🔹 MODEL ROUTING
    query_type = classify_query(prompt)

    if query_type == "documentation":
        selected_model = MODEL_DOC
    elif query_type == "reasoning":
        selected_model = MODEL_REASONING
    elif query_type == "pql_generation":
        selected_model = MODEL_PQL_ENGINE
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
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
