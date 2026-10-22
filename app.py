import os
import json
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

# Target directory for uploaded files
UPLOAD_DIR = "data/raw_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("RAG Intelligence Studio")
st.caption("Deterministic document search & benchmarked RRF ensemble powered by Ollama")

# Sidebar - Configuration & Document Management
with st.sidebar:
    st.header("Pipeline Configuration")
    st.caption("Manage document corpus & hybrid search parameters.")
    
    st.subheader("Upload PDF Documents")
    uploaded_pdfs = st.file_uploader(
        "Upload additional PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Save newly uploaded files to local disk
    if uploaded_pdfs:
        for uploaded_file in uploaded_pdfs:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_pdfs)} PDF(s).")
    
    st.markdown("---")
    st.subheader("Document Corpus")
    
    # Automatically scan data/raw_documents folder for all PDFs
    pdf_files = [
        os.path.join(UPLOAD_DIR, f) 
        for f in os.listdir(UPLOAD_DIR) 
        if f.endswith(".pdf")
    ]
    
    if pdf_files:
        for pdf in pdf_files:
            filename = os.path.basename(pdf)
            st.markdown(f"• `{filename}`")
    else:
        st.info("No PDF documents found in corpus.")
        
    st.markdown("---")
    top_k = st.slider("Top-k Depth", min_value=5, max_value=25, value=15)
    
    if st.button("⚡ Build Vector & BM25 Index", use_container_width=True):
        if not pdf_files:
            st.error("Please upload at least one PDF or ensure data/raw_documents/ contains PDFs.")
        else:
            with st.spinner("Processing documents & building hybrid index..."):
                docs = load_and_chunk_documents(pdf_files)
                st.session_state.ensemble_retriever = build_hybrid_retriever(docs, top_k=top_k)
                
                if hasattr(st.session_state.ensemble_retriever, 'retrievers'):
                    st.session_state.faiss_retriever = st.session_state.ensemble_retriever.retrievers[0]
                else:
                    st.session_state.faiss_retriever = st.session_state.ensemble_retriever.dense_retriever
                    
                st.session_state.rag_chain = build_conversational_rag_chain(st.session_state.ensemble_retriever)
                st.success("Index Built Successfully!")

# Main Workspace Tabs
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
    st.caption("Compare Dense Vector Search (FAISS) against Hybrid Ensemble Search (Dense + BM25) across all ground-truth evaluation questions.")
    
    run_bench = st.button("Run Evaluation Benchmark", type="primary")
    
    if run_bench:
        if "ensemble_retriever" in st.session_state:
            with st.spinner("Running evaluation grid..."):
                results = run_benchmark_evaluation(
                    st.session_state.ensemble_retriever,
                    st.session_state.faiss_retriever,
                    questions_json_path="data/ground_truth.json"
                )
                st.session_state.latest_results = results
        else:
            st.warning("Please click '⚡ Build Vector & BM25 Index' in the sidebar first.")

    display_results = getattr(st.session_state, 'latest_results', None)
    if not display_results and os.path.exists("results.json"):
        try:
            with open("results.json", "r") as f:
                display_results = json.load(f)
        except Exception:
            pass

    if display_results:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DENSE HIT RATE@15", f"{display_results.get('dense_hit_rate', 0)*100:.2f}%")
        col2.metric("DENSE MRR@15", f"{display_results.get('dense_mrr', 0):.3f}")
        col3.metric("HYBRID HIT RATE@15", f"{display_results.get('hybrid_hit_rate', 0)*100:.2f}%")
        col4.metric("HYBRID MRR@15", f"{display_results.get('hybrid_mrr', 0):.3f}")