import os
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field


class SimpleHybridEnsembleRetriever(BaseRetriever):
    """Custom Hybrid Ensemble Retriever using Reciprocal Rank Fusion (RRF)."""

    bm25_retriever: Any = Field(...)
    dense_retriever: Any = Field(...)
    c: int = Field(default=60)
    k: int = Field(default=8)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        rrf_scores = {}
        doc_map = {}

        # Process BM25 rankings
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = doc.metadata.get(
                "chunk_id", hash(doc.page_content)
            )
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (self.c + rank)
            )

        # Process Dense Vector rankings
        for rank, doc in enumerate(dense_docs, start=1):
            doc_id = doc.metadata.get(
                "chunk_id", hash(doc.page_content)
            )
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (self.c + rank)
            )

        # Sort documents by RRF score descending
        sorted_doc_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[: self.k]]


def build_retrievers(
    chunks: List[Document], k: int = 8
) -> Tuple[Any, SimpleHybridEnsembleRetriever]:
    """Builds Dense (FAISS) and Hybrid (FAISS + BM25) retrievers locally."""
    print("Loading local BGE embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
    )

    # 1. Dense Vector Store (FAISS)
    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 2. Sparse Keyword Search (BM25)
    print("Building BM25 keyword index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 3. Custom Hybrid Ensemble Retriever
    print("Combining into Hybrid Ensemble Retriever...")
    ensemble_retriever = SimpleHybridEnsembleRetriever(
        bm25_retriever=bm25_retriever, dense_retriever=dense_retriever, k=k
    )

    return dense_retriever, ensemble_retriever