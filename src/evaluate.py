import json
import streamlit as st
from typing import List, Dict, Any


def normalize_meta_str(text: Any) -> str:
    """Standardizes document titles for clean matching."""
    if not text:
        return ""
    return str(text).lower().replace(" ", "").replace("_", "").replace("-", "")


def is_page_match(doc_metadata: dict, expected_page: int) -> bool:
    """
    Checks if retrieved document metadata page matches expected ground truth page.
    Handles both 0-indexed and 1-indexed PyPDF page numbers.
    """
    if "page" not in doc_metadata:
        return False
    
    retrieved_page = doc_metadata["page"]
    
    # Check exact match, 1-based index conversion, or string matching
    try:
        page_num = int(retrieved_page)
        exp_num = int(expected_page)
        return page_num == exp_num or page_num + 1 == exp_num
    except (ValueError, TypeError):
        return str(retrieved_page).strip() == str(expected_page).strip()


def calculate_metrics(retrieved_docs: List[Any], expected_doc: str, expected_page: int) -> tuple[float, float]:
    """
    Calculates Hit Rate and MRR by checking if the expected page and document 
    are present in the top-k retrieved chunks.
    """
    norm_exp_doc = normalize_meta_str(expected_doc)

    for rank, doc in enumerate(retrieved_docs, start=1):
        meta = getattr(doc, "metadata", {})
        
        # Check source document matching (if metadata source is available)
        doc_source = meta.get("source", "") or meta.get("file_name", "")
        norm_source = normalize_meta_str(doc_source)
        
        doc_matches = True
        if norm_exp_doc and norm_source:
            doc_matches = (norm_exp_doc in norm_source) or (norm_source in norm_exp_doc)

        # Check page number matching
        page_matches = is_page_match(meta, expected_page)

        if page_matches and doc_matches:
            return 1.0, 1.0 / rank

    return 0.0, 0.0


def run_benchmark_evaluation(ensemble_retriever, faiss_retriever, questions_json_path: str = "data/ground_truth.json") -> Dict[str, Any]:
    """
    Runs evaluation benchmark across all ground-truth questions based on document metadata.
    """
    with open(questions_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    dense_hits, hybrid_hits = [], []
    dense_mrrs, hybrid_mrrs = [], []
    detailed_logs = []

    for idx, item in enumerate(ground_truth_data):
        question = item.get("question", "")
        expected_doc = item.get("expected_doc", "")
        expected_page = item.get("expected_page", 0)

        # 1. Retrieve Chunks
        dense_docs = faiss_retriever.invoke(question)
        hybrid_docs = ensemble_retriever.invoke(question)

        # 2. Evaluate Metadata Page Matches
        d_hit, d_mrr = calculate_metrics(dense_docs, expected_doc, expected_page)
        h_hit, h_mrr = calculate_metrics(hybrid_docs, expected_doc, expected_page)

        dense_hits.append(d_hit)
        dense_mrrs.append(d_mrr)
        hybrid_hits.append(h_hit)
        hybrid_mrrs.append(h_mrr)

        detailed_logs.append({
            "id": item.get("id"),
            "question": question,
            "expected_page": expected_page,
            "dense_hit": bool(d_hit),
            "hybrid_hit": bool(h_hit),
            "dense_mrr": d_mrr,
            "hybrid_mrr": h_mrr
        })

    summary_results = {
        "dense_hit_rate": sum(dense_hits) / len(dense_hits) if dense_hits else 0.0,
        "hybrid_hit_rate": sum(hybrid_hits) / len(hybrid_hits) if hybrid_hits else 0.0,
        "dense_mrr": sum(dense_mrrs) / len(dense_mrrs) if dense_mrrs else 0.0,
        "hybrid_mrr": sum(hybrid_mrrs) / len(hybrid_mrrs) if hybrid_mrrs else 0.0,
        "total_questions": len(ground_truth_data),
        "logs": detailed_logs
    }

    with open("results.json", "w") as f:
        json.dump(summary_results, f, indent=4)

    return summary_results