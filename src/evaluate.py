import json
import os
from typing import List, Dict, Any


def evaluate_retrieval(
    retriever,
    ground_truth_file: str = "data/ground_truth.json",
    k: int = 8,
    output_file: str = None
) -> Dict[str, Any]:
    """Calculates Retrieval Hit Rate@k and MRR@k against ground truth dataset and optionally saves results to JSON."""
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    hits = 0
    reciprocal_ranks = []
    results_detail = []

    for item in ground_truth:
        q_id = item["id"]
        question = item["question"]
        expected_sources = item.get("expected_source_chunk_ids", [])

        # Retrieve top k documents
        retrieved_docs = retriever.invoke(question)
        retrieved_chunk_ids = [
            doc.metadata.get("chunk_id", "") for doc in retrieved_docs[:k]
        ]

        is_hit = False
        rank = 0

        # Match retrieved chunk against ground truth expected sources
        for idx, r_cid in enumerate(retrieved_chunk_ids, start=1):
            for e_src in expected_sources:
                if e_src.lower() in r_cid.lower():
                    if not is_hit:
                        is_hit = True
                        rank = idx
                    break
            if is_hit:
                break

        if is_hit:
            hits += 1

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        results_detail.append(
            {
                "question_id": q_id,
                "question": question,
                "hit": is_hit,
                "rank": rank,
                "reciprocal_rank": rr,
                "retrieved_chunks": retrieved_chunk_ids,
            }
        )

    total_questions = len(ground_truth)
    hit_rate = (hits / total_questions) * 100 if total_questions > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total_questions if total_questions > 0 else 0.0

    metrics_summary = {
        "hit_rate_at_k": round(hit_rate, 2),
        "mrr_at_k": round(mrr, 3),
        "k": k,
        "total_questions": total_questions,
        "details": results_detail,
    }

    # Save to output file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(metrics_summary, out_f, indent=2)
        print(f"Evaluation results successfully saved to: {output_file}")

    return metrics_summary