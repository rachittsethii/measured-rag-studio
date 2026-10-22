import json
from src.loader import load_and_chunk_docs
from src.retriever import build_retrievers
from src.evaluate import evaluate_retrieval
from src.chain import build_rag_chain

def main():
    print("=== STEP 1: Loading & Chunking Documents ===")
    chunks = load_and_chunk_docs()

    print("\n=== STEP 2: Building Dense & Hybrid Retrievers ===")
    dense_retriever, hybrid_retriever = build_retrievers(chunks, k=8)

    print("\n=== STEP 3: Evaluating Dense Vector Retriever ===")
    dense_metrics = evaluate_retrieval(dense_retriever, k=8)
    print(f"Dense Hit Rate@8: {dense_metrics['hit_rate_at_k']}% | MRR@8: {dense_metrics['mrr_at_k']}")

    print("\n=== STEP 4: Evaluating Hybrid Ensemble Retriever ===")
    hybrid_metrics = evaluate_retrieval(hybrid_retriever, k=8, output_file="results.json")
    print(f"Hybrid Hit Rate@8: {hybrid_metrics['hit_rate_at_k']}% | MRR@8: {hybrid_metrics['mrr_at_k']}")

    # Save combined comparison summary to results.json
    combined_results = {
        "dense_vector_retriever": {
            "hit_rate_at_8": dense_metrics['hit_rate_at_k'],
            "mrr_at_8": dense_metrics['mrr_at_k']
        },
        "hybrid_ensemble_retriever": {
            "hit_rate_at_8": hybrid_metrics['hit_rate_at_k'],
            "mrr_at_8": hybrid_metrics['mrr_at_k'],
            "details": hybrid_metrics['details']
        }
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    print("Combined comparison metrics exported to results.json\n")

    print("=== STEP 5: Testing RAG Chain with Citations ===")
    rag_chain = build_rag_chain(hybrid_retriever)

    sample_question = "What are the key cyber resilience requirements defined in CSCRF?"
    print(f"\nQuestion: {sample_question}\n")

    response = rag_chain.invoke(sample_question)
    print(f"Answer with Citations:\n{response}\n")

if __name__ == "__main__":
    main()