import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_documents(file_paths: List[str]) -> List[Document]:
    """
    Loads PDF documents and splits them into granular chunks preserving page metadata.
    """
    all_docs = []
    
    for path in file_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            loaded_pages = loader.load()
            all_docs.extend(loaded_pages)
            
    # Tighter chunking for optimal hybrid vector & keyword lookup
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=125,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(all_docs)