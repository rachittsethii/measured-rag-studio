from typing import List, Any
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class CustomEnsembleRetriever:
    """
    Standalone Hybrid Ensemble Retriever implementing Reciprocal Rank Fusion (RRF).
    Combines Dense Vector Search (FAISS) and Sparse Keyword Search (BM25).
    """
    def __init__(self, dense_retriever: Any, sparse_retriever: Any, c: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.retrievers = [dense_retriever, sparse_retriever]
        self.c = c

    def get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        sparse_docs = self.sparse_retriever.get_relevant_documents(query)
        
        doc_scores = {}
        
        # Process Dense Docs
        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 0.5 * (1.0 / (self.c + rank))
            
        # Process Sparse Docs
        for rank, doc in enumerate(sparse_docs, start=1):
            key = doc.page_content
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 0.5 * (1.0 / (self.c + rank))
            
        # Sort by fused RRF score descending
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


@st.cache_resource
def get_embeddings_model():
    """Cache embedding model weights in memory."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )


def build_hybrid_retriever(docs: List[Document], top_k: int = 12) -> CustomEnsembleRetriever:
    """
    Builds a Hybrid Ensemble Retriever using FAISS and BM25.
    """
    embeddings = get_embeddings_model()
    
    # 1. Dense Vector Store (FAISS)
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # 2. Sparse Keyword Store (BM25 with custom params)
    bm25_retriever = BM25Retriever.from_documents(
        docs, 
        k=top_k, 
        bm25_params={"k1": 1.2, "b": 0.75}
    )
    
    return CustomEnsembleRetriever(dense_retriever=faiss_retriever, sparse_retriever=bm25_retriever)