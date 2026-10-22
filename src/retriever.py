from typing import List
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever


@st.cache_resource
def get_embeddings_model():
    """Cache embedding model weights in memory."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )


def build_hybrid_retriever(docs: List[Document], top_k: int = 12) -> EnsembleRetriever:
    """
    Builds a Hybrid Ensemble Retriever combining Dense FAISS Vector Search 
    and Sparse BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
    """
    embeddings = get_embeddings_model()
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k1 = 1.2
    bm25_retriever.b = 0.75
    bm25_retriever.k = top_k
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever