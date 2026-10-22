import os
import streamlit as st
from src.loader import load_and_chunk_documents
from src.retriever import build_hybrid_retriever
from src.evaluate import run_benchmark_evaluation
from src.chain import build_conversational_rag_chain

st.set_page_config(
    page_title="RAG Intelligence Studio",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark UI & Clean Metric Cards
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
    }
    .status-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("RAG Intelligence Studio")
st.caption("Deterministic document search & benchmarked RRF ensemble powered by Ollama")

# Sidebar - Configuration & Document Corpus
with st.sidebar:
    st.header("Pipeline Configuration")
    st.caption("Manage document corpus & hybrid search parameters.")
    
    # Raw PDF Corpus List
    pdf_files = [
        "data/raw_documents/Cybersecurity and Cyber Resilience Framework (CSCRF) for SEBI Regulated Entities (REs)_1724679211.pdf",
        "data/raw_documents/Presentation - CSCRF.pdf",
        "data/raw_documents/google_privacy_policy_en.pdf"
    ]
    
    st.markdown("---")
    st.subheader("Document Corpus")
    for pdf in pdf_files:
        filename = os.path.basename(pdf)
        st.markdown(f"• `{filename}`")
        
    st.markdown("---")
    top_k = st.slider("Top-k Depth", min_value=5, max_value=25, value=15)
    
    if st.button("⚡ Build Vector & BM25 Index", use_container_width=True):
        with st.spinner("Processing documents & building hybrid index..."):
            docs = load_and_chunk_documents(pdf_files)
            st.session_state.ensemble_retriever = build_hybrid_retriever(docs, top_k=top_k)
            # Access underlying FAISS retriever from custom/standard ensemble
            if hasattr(st.session_state.ensemble_retriever, 'retrievers'):
                st.session_state.faiss_retriever = st.session_state.ensemble_retriever.retrievers[0]
            else:
                st.session_state.faiss_retriever = st.session_state.ensemble_retriever.dense_retriever
                
            st.session_state.rag_chain = build_conversational_rag_chain(st.session_state.ensemble_retriever)
            st.success("Index Built Successfully!")

# Main Tabs
tab1, tab2 = st.tabs(["💬 Conversational Workspace", "📊 Retrieval Benchmark & Analytics"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input("Ask a question about your indexed documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        if "rag_chain" in st.session_state:
            try:
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": []
                })
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"Execution Error: {e}")
        else:
            st.warning("Please click '⚡ Build Vector & BM25 Index' in the sidebar first.")

with tab2:
    st.subheader("Quantitative Retrieval Benchmarks")
    st.caption("Compare Dense Vector Search (FAISS) against Hybrid Ensemble Search (Dense + BM25) across all 30 ground-truth evaluation questions.")
    
    if st.button("Run Evaluation Benchmark", type="primary"):
        if "ensemble_retriever" in st.session_state:
            with st.spinner("Running 30-question evaluation grid..."):
                results = run_benchmark_evaluation(
                    st.session_state.ensemble_retriever,
                    st.session_state.faiss_retriever,
                    questions_json_path="data/ground_truth.json"
                )
                
                # Render 4 Benchmark Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("DENSE HIT RATE@15", f"{results['dense_hit_rate']*100:.2f}%")
                col2.metric("DENSE MRR@15", f"{results['dense_mrr']:.3f}")
                col3.metric("HYBRID HIT RATE@15", f"{results['hybrid_hit_rate']*100:.2f}%")
                col4.metric("HYBRID MRR@15", f"{results['hybrid_mrr']:.3f}")
                
                st.success(f"Benchmark completed across {results.get('total_questions', 30)} questions! Detailed log exported to results.json.")
        else:
            st.warning("Please click '⚡ Build Vector & BM25 Index' in the sidebar first.")