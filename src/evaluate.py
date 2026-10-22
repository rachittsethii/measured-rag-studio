import json
import os
from typing import List, Dict, Any

def evaluate_retrieval(
    retriever: Any,
    ground_truth_file: str = "data/ground_truth.json",
    k: int = 8,
    output_file: str = None
) -> Dict[str, Any]:
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground truth file not found at {ground_truth_file}")

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    hits = 0
    mrr_sum = 0.0
    details = []

    for item in ground_truth:
        query = item["question"]
        target_doc = item.get("expected_doc", "").lower()
        target_page = item.get("expected_page", None)
        target_chunk_id = item.get("expected_chunk_id", None)

        retrieved_docs = retriever.invoke(query)[:k]
        
        hit_found = False
        rank_found = 0

        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_source = str(doc.metadata.get("source", "")).lower()
            doc_page = doc.metadata.get("page", None)
            chunk_id = doc.metadata.get("chunk_id", "")

            # Flexible evaluation logic: exact chunk ID match OR document + page match
            is_page_match = (target_doc in doc_source) and (target_page is not None and doc_page == target_page)
            is_chunk_match = (target_chunk_id is not None and chunk_id == target_chunk_id)

            if is_chunk_match or is_page_match:
                hit_found = True
                rank_found = rank
                break

        if hit_found:
            hits += 1
            mrr_sum += 1.0 / rank_found

        details.append({
            "question": query,
            "hit": hit_found,
            "rank": rank_found if hit_found else None,
            "retrieved_sources": [
                f"{os.path.basename(doc.metadata.get('source', ''))}_p{doc.metadata.get('page', '')}" 
                for doc in retrieved_docs
            ]
        })

    total_queries = len(ground_truth)
    hit_rate = round((hits / total_queries) * 100, 2) if total_queries > 0 else 0.0
    mrr = round(mrr_sum / total_queries, 3) if total_queries > 0 else 0.0

    summary = {
        "total_queries": total_queries,
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": mrr,
        "details": details
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary