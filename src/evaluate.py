import json
import re
import difflib
import streamlit as st
from typing import List, Dict, Any


def clean_str(val: Any) -> str:
    """Safely converts any JSON element or list into a plain text string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join([clean_str(x) for x in val])
    if isinstance(val, dict):
        return " ".join([clean_str(v) for v in val.values()])
    return str(val)


def normalize_text(text: str) -> str:
    """Removes special characters, extra whitespace, and converts to lowercase."""
    text = clean_str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def calculate_metrics(retrieved_docs: List[Any], ground_truth_text: str) -> tuple[float, float]:
    """
    Calculates Hit Rate and MRR using multi-tier matching:
    1. Substring Match
    2. Token Set Overlap (>= 20% key words)
    3. Fuzzy Sequence Similarity (>= 0.35 ratio)
    """
    norm_gt = normalize_text(ground_truth_text)
    if not norm_gt:
        return 0.0, 0.0

    gt_words = set(norm_gt.split())

    for rank, doc in enumerate(retrieved_docs, start=1):
        norm_doc = normalize_text(doc.page_content)
        if not norm_doc:
            continue

        # Tier 1: Substring Match
        if norm_gt in norm_doc or norm_doc in norm_gt:
            return 1.0, 1.0 / rank

        # Tier 2: Token Overlap Match
        doc_words = set(norm_doc.split())
        if gt_words and doc_words:
            overlap = gt_words.intersection(doc_words)
            if len(gt_words) > 0 and (len(overlap) / len(gt_words)) >= 0.20:
                return 1.0, 1.0 / rank

        # Tier 3: Fuzzy Ratio
        similarity = difflib.SequenceMatcher(None, norm_gt[:200], norm_doc[:200]).ratio()
        if similarity >= 0.35:
            return 1.0, 1.0 / rank

    return 0.0, 0.0


def run_benchmark_evaluation(ensemble_retriever, faiss_retriever, questions_json_path: str = "data/ground_truth.json") -> Dict[str, Any]:
    """
    Runs evaluation benchmark across all ground-truth questions and prints debug diagnostics.
    """
    with open(questions_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    dense_hits, hybrid_hits = [], []
    dense_mrrs, hybrid_mrrs = [], []
    detailed_logs = []

    for idx, item in enumerate(ground_truth_data):
        # Broad JSON key extraction
        question = clean_str(item.get("question") or item.get("query") or item.get("q"))
        gt_context = (
            item.get("ground_truth_context") or
            item.get("ground_truth") or
            item.get("context") or
            item.get("answer") or
            item.get("expected_text") or
            ""
        )
        gt_context_str = clean_str(gt_context)

        # 1. Retrieve Docs
        dense_docs = faiss_retriever.invoke(question)
        hybrid_docs = ensemble_retriever.invoke(question)

        # Print Debug Info for Question #1 on screen
        if idx == 0:
            st.info(f"🔍 **Debug Question #1:** {question}")
            st.info(f"📄 **Expected Context:** {gt_context_str[:150]}...")
            if hybrid_docs:
                st.info(f"⚡ **First Retrieved Chunk:** {hybrid_docs[0].page_content[:150]}...")

        # 2. Calculate Metrics
        d_hit, d_mrr = calculate_metrics(dense_docs, gt_context_str)
        h_hit, h_mrr = calculate_metrics(hybrid_docs, gt_context_str)

        dense_hits.append(d_hit)
        dense_mrrs.append(d_mrr)
        hybrid_hits.append(h_hit)
        hybrid_mrrs.append(h_mrr)

        detailed_logs.append({
            "question": question,
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