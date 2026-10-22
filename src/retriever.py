from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Safe import for EnsembleRetriever across different LangChain versions
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever


def build_hybrid_retriever(docs: List[Document], top_k: int = 12) -> EnsembleRetriever:
    """
    Builds a Hybrid Ensemble Retriever combining Dense FAISS Vector Search 
    and Sparse BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
    """
    # 1. Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Dense Vector Store (FAISS)
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # 3. Sparse Keyword Store (BM25 with tuned parameters)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k1 = 1.2
    bm25_retriever.b = 0.75
    bm25_retriever.k = top_k
    
    # 4. Hybrid Ensemble Retriever (50/50 Weighting)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever