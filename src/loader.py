import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Loads PDF documents and splits them into granular chunks optimized for 
    high-precision retrieval benchmarks.
    """
    all_docs = []
    
    for path in pdf_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
            
    # Tighter chunking strategy for precise regulatory and policy Q&A matching
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(all_docs)
    return chunked_docs