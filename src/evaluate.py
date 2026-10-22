import json
import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """Removes special characters, extra whitespaces, and converts to lowercase."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def calculate_metrics(retrieved_docs: List[Any], ground_truth_text: str) -> tuple[float, float]:
    """
    Calculates Hit Rate and Reciprocal Rank (MRR) using normalized substring matching.
    """
    norm_gt = normalize_text(ground_truth_text)
    
    hit = 0.0
    reciprocal_rank = 0.0
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        norm_doc = normalize_text(doc.page_content)
        if norm_gt in norm_doc or norm_doc in norm_gt:
            hit = 1.0
            reciprocal_rank = 1.0 / rank
            break
            
    return hit, reciprocal_rank


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
        question = item["question"]
        gt_context = item["ground_truth_context"]
        
        # 1. Evaluate Dense FAISS (using .invoke)
        dense_docs = faiss_retriever.invoke(question)
        d_hit, d_mrr = calculate_metrics(dense_docs, gt_context)
        dense_hits.append(d_hit)
        dense_mrrs.append(d_mrr)
        
        # 2. Evaluate Hybrid Ensemble (using .invoke)
        hybrid_docs = ensemble_retriever.invoke(question)
        h_hit, h_mrr = calculate_metrics(hybrid_docs, gt_context)
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
        "dense_hit_rate": sum(dense_hits) / len(dense_hits),
        "hybrid_hit_rate": sum(hybrid_hits) / len(hybrid_hits),
        "dense_mrr": sum(dense_mrrs) / len(dense_mrrs),
        "hybrid_mrr": sum(hybrid_mrrs) / len(hybrid_mrrs),
        "total_questions": len(ground_truth_data),
        "logs": detailed_logs
    }
    
    with open("results.json", "w") as f:
        json.dump(summary_results, f, indent=4)
        
    return summary_results