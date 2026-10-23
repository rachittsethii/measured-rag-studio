import json
import re
import difflib
import streamlit as st
from typing import List, Dict, Any


def clean_str(val: Any) -> str:
    """Helper to safely convert any JSON type (list, dict, int) to clean text."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join([clean_str(x) for x in val])
    if isinstance(val, dict):
        return " ".join([clean_str(v) for v in val.values()])
    return str(val)


def normalize_text(text: str) -> str:
    """Removes non-alphanumeric characters and converts to lowercase."""
    text = clean_str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def calculate_metrics(retrieved_docs: List[Any], ground_truth_text: str) -> tuple[float, float]:
    """
    Calculates Hit Rate and MRR using multi-strategy matching:
    1. Direct Substring Match
    2. Word Set Overlap (>= 30% key terms)
    3. Fuzzy Sequence Similarity (Ratio >= 0.45)
    """
    norm_gt = normalize_text(ground_truth_text)
    if not norm_gt:
        return 0.0, 0.0

    gt_words = set(norm_gt.split())
    hit = 0.0
    reciprocal_rank = 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        norm_doc = normalize_text(doc.page_content)
        if not norm_doc:
            continue

        # Strategy 1: Substring Match
        if norm_gt in norm_doc or norm_doc in norm_gt:
            return 1.0, 1.0 / rank

        # Strategy 2: Significant Word Overlap
        doc_words = set(norm_doc.split())
        if gt_words and doc_words:
            overlap = gt_words.intersection(doc_words)
            if len(gt_words) > 0 and (len(overlap) / len(gt_words)) >= 0.30:
                return 1.0, 1.0 / rank

        # Strategy 3: Fuzzy Sequence Ratio
        similarity = difflib.SequenceMatcher(None, norm_gt[:300], norm_doc[:300]).ratio()
        if similarity >= 0.40:
            return 1.0, 1.0 / rank

    return hit, reciprocal_rank


def run_benchmark_evaluation(ensemble_retriever, faiss_retriever, questions_json_path: str = "data/ground_truth.json") -> Dict[str, Any]:
    """
    Runs evaluation benchmark across all ground-truth questions safely.
    """
    with open(questions_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    dense_hits, hybrid_hits = [], []
    dense_mrrs, hybrid_mrrs = [], []
    detailed_logs = []

    for idx, item in enumerate(ground_truth_data):
        # Extract question
        question = clean_str(item.get("question") or item.get("query") or item.get("q"))

        # Extract context using broad field detection
        gt_context = (
            item.get("ground_truth_context") or
            item.get("ground_truth") or
            item.get("context") or
            item.get("answer") or
            item.get("expected_text") or
            item.get("text") or
            ""
        )
        gt_context_str = clean_str(gt_context)

        # 1. Evaluate Dense FAISS
        dense_docs = faiss_retriever.invoke(question)
        d_hit, d_mrr = calculate_metrics(dense_docs, gt_context_str)
        dense_hits.append(d_hit)
        dense_mrrs.append(d_mrr)

        # 2. Evaluate Hybrid Ensemble
        hybrid_docs = ensemble_retriever.invoke(question)
        h_hit, h_mrr = calculate_metrics(hybrid_docs, gt_context_str)
        hybrid_hits.append(h_hit)
        hybrid_mrrs.append(h_mrr)

        detailed_logs.append({
            "question": question,
            "gt_snippet": gt_context_str[:60],
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