import os
import json
import streamlit as st
from src.loader import load_and_chunk_docs
from src.retriever import build_retrievers
from src.evaluate import evaluate_retrieval
from src.chain import build_conversational_rag_chain

st.set_page_config(
    page_title="RAG Evaluation & Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PROFESSIONAL FRONTEND CSS INJECTION (Tailwind-like Dark Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Body styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Glassmorphism Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Custom Header Banner */
    .hero-container {
        padding: 24px 0 16px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #22c55e;
        box-shadow: 0 0 8px #22c55e;
    }

    /* Metrics Dashboard Glass Cards */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 20px 35px -10px rgba(124, 58, 237, 0.25);
    }

    .stat-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .stat-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-accent {
        color: #a78bfa;
    }

    /* Custom Chat Message Containers */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
    }

    /* Custom Input Controls & Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px 0 rgba(139, 92, 246, 0.6) !important;
    }

    /* Tabs Bar Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #818cf8 !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BACKEND PATHS & SESSION INITIALIZATION
# ---------------------------------------------------------
RAW_DOCS_DIR = "data/raw_documents"
GROUND_TRUTH_FILE = "data/ground_truth.json"
os.makedirs(RAW_DOCS_DIR, exist_ok=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

@st.cache_resource(show_spinner=False)
def get_cached_retrievers(doc_dir: str, k: int):
    chunks = load_and_chunk_docs(doc_dir=doc_dir)
    dense, hybrid = build_retrievers(chunks, k=k)
    return chunks, dense, hybrid

# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    st.caption("Manage document corpus & hybrid search parameters.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(RAW_DOCS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} PDF(s).")
        st.cache_resource.clear()

    existing_files = os.listdir(RAW_DOCS_DIR)
    st.markdown("#### 📄 Document Corpus")
    if existing_files:
        for fname in existing_files:
            st.markdown(f"• `{fname}`")
    else:
        st.info("No documents found in storage.")

    st.divider()
    k_val = st.slider("Top-k Depth", min_value=2, max_value=12, value=8, step=1)

    if st.button("⚡ Build Vector & BM25 Index"):
        with st.spinner("Processing embeddings & constructing hybrid indices..."):
            try:
                chunks, dense, hybrid = get_cached_retrievers(doc_dir=RAW_DOCS_DIR, k=k_val)
                rag_chain = build_conversational_rag_chain(hybrid)

                st.session_state["chunks"] = chunks
                st.session_state["dense_retriever"] = dense
                st.session_state["hybrid_retriever"] = hybrid
                st.session_state["rag_chain"] = rag_chain
                st.session_state["indexed"] = True
                st.success("Indexing Complete!")
            except Exception as e:
                st.error(f"Indexing Error: {e}")

# ---------------------------------------------------------
# MAIN DASHBOARD HERO SECTION
# ---------------------------------------------------------
st.markdown('''
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="hero-title">RAG Intelligence Studio</div>
            <div class="hero-subtitle">Deterministic document search & benchmarked RRF ensemble powered by Ollama</div>
        </div>
        <div>
            <span class="status-badge"><span class="status-dot"></span> System Operational</span>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Conversational Workspace", "📊 Retrieval Benchmark & Analytics"])

# ---------------------------------------------------------
# TAB 1: CONVERSATIONAL QA INTERFACE
# ---------------------------------------------------------
with tab1:
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your indexed documents..."):
        if not st.session_state.get("indexed", False):
            st.warning("Please index your documents using the sidebar control panel first.")
        else:
            st.chat_message("user").markdown(user_query)

            formatted_history = []
            for msg in st.session_state["chat_history"]:
                role = "human" if msg["role"] == "user" else "ai"
                formatted_history.append((role, msg["content"]))

            with st.spinner("Retrieving relevant context & reasoning..."):
                try:
                    chain = st.session_state["rag_chain"]
                    response = chain.invoke({
                        "input": user_query,
                        "chat_history": formatted_history
                    })

                    with st.chat_message("assistant"):
                        st.markdown(response)

                    st.session_state["chat_history"].append({"role": "user", "content": user_query})
                    st.session_state["chat_history"].append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Execution Error: {e}")

# ---------------------------------------------------------
# TAB 2: METRICS & BENCHMARK DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📊 Quantitative Retrieval Benchmarks")
    st.caption("Compare dense vector search against the hybrid RRF ensemble using `data/ground_truth.json`.")

    if st.button("Run Evaluation Benchmark"):
        if not st.session_state.get("indexed", False):
            st.warning("Please index your documents first.")
        else:
            with st.spinner("Evaluating retrieval pipelines..."):
                dense = st.session_state["dense_retriever"]
                hybrid = st.session_state["hybrid_retriever"]

                dense_res = evaluate_retrieval(dense, ground_truth_file=GROUND_TRUTH_FILE, k=k_val)
                hybrid_res = evaluate_retrieval(hybrid, ground_truth_file=GROUND_TRUTH_FILE, k=k_val, output_file="results.json")

                # Custom High-End Metric Display
                st.markdown(f'''
                <div class="card-grid">
                    <div class="stat-card">
                        <div class="stat-label">Dense Hit Rate@{k_val}</div>
                        <div class="stat-value">{dense_res["hit_rate_at_k"]}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Dense MRR@{k_val}</div>
                        <div class="stat-value">{dense_res["mrr_at_k"]}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Hybrid Hit Rate@{k_val}</div>
                        <div class="stat-value stat-accent">{hybrid_res["hit_rate_at_k"]}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Hybrid MRR@{k_val}</div>
                        <div class="stat-value stat-accent">{hybrid_res["mrr_at_k"]}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.success("Benchmark completed! Detailed performance log saved to `results.json`.")

                with st.expander("🔎 Question-Level Detailed Log"):
                    st.json(hybrid_res["details"])