import json
import streamlit as st
from typing import List, Dict, Any


def is_page_match_window(retrieved_page: Any, expected_page: Any, window: int = 3) -> bool:
    """
    Checks if retrieved page falls within a ±window offset of expected page.
    Handles 0-indexed PyPDF loaders vs 1-indexed document page numbering.
    """
    try:
        r_page = int(retrieved_page)
        e_page = int(expected_page)
        
        # Direct match, 1-based offset, or ±window range
        return abs(r_page - e_page) <= window or abs((r_page + 1) - e_page) <= window
    except (ValueError, TypeError):
        return str(retrieved_page).strip() == str(expected_page).strip()


def calculate_metrics(retrieved_docs: List[Any], expected_doc: str, expected_page: int) -> tuple[float, float]:
    """
    Calculates Hit Rate and MRR across retrieved chunks.
    """
    exp_doc_clean = str(expected_doc).lower().replace(" ", "").replace("-", "").replace("_", "")

    for rank, doc in enumerate(retrieved_docs, start=1):
        meta = getattr(doc, "metadata", {})
        
        # Extract metadata fields
        retrieved_page = meta.get("page", -999)
        doc_source = str(meta.get("source", "") or meta.get("file_name", "")).lower().replace(" ", "").replace("-", "").replace("_", "")
        
        # Document title match check
        doc_match = True
        if exp_doc_clean and doc_source:
            doc_match = (exp_doc_clean in doc_source) or (doc_source in exp_doc_clean)
            
        # Page match check
        page_match = is_page_match_window(retrieved_page, expected_page, window=3)

        if page_match and doc_match:
            return 1.0, 1.0 / rank

    return 0.0, 0.0


def run_benchmark_evaluation(ensemble_retriever, faiss_retriever, questions_json_path: str = "data/ground_truth.json") -> Dict[str, Any]:
    """
    Runs evaluation benchmark across all ground-truth questions.
    """
    with open(questions_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    dense_hits, hybrid_hits = [], []
    dense_mrrs, hybrid_mrrs = [], []
    detailed_logs = []

    for item in ground_truth_data:
        question = item.get("question", "")
        expected_doc = item.get("expected_doc", "")
        expected_page = item.get("expected_page", 0)

        # 1. Retrieve Chunks
        dense_docs = faiss_retriever.invoke(question)
        hybrid_docs = ensemble_retriever.invoke(question)

        # 2. Evaluate Window Match
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