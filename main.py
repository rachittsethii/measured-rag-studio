import json
import os
from src.loader import load_and_chunk_docs
from src.retriever import build_retrievers
from src.evaluate import evaluate_retrieval

def run_grid_experiment(chunk_size: int, chunk_overlap: int, k: int = 8, ground_truth_path: str = "data/ground_truth.json"):
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENT: Chunk Size = {chunk_size} | Overlap = {chunk_overlap} | Top-k = {k}")
    print(f"=======================================================")
    
    # 1. Load documents with targeted chunk configuration
    chunks = load_and_chunk_docs(
        doc_dir="data/raw_documents",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Generated {len(chunks)} chunks.")

    # 2. Build FAISS Dense Vector & BM25 Hybrid Retrievers
    dense_retriever, hybrid_retriever = build_retrievers(chunks, k=k)

    # 3. Evaluate both strategies on the 30-question ground truth dataset
    print("Evaluating Dense FAISS Retriever...")
    dense_metrics = evaluate_retrieval(dense_retriever, ground_truth_file=ground_truth_path, k=k)

    print("Evaluating Hybrid Ensemble (Dense + BM25 RRF)...")
    hybrid_metrics = evaluate_retrieval(hybrid_retriever, ground_truth_file=ground_truth_path, k=k)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "dense": {
            "hit_rate_at_k": dense_metrics["hit_rate_at_k"],
            "mrr_at_k": dense_metrics["mrr_at_k"]
        },
        "hybrid": {
            "hit_rate_at_k": hybrid_metrics["hit_rate_at_k"],
            "mrr_at_k": hybrid_metrics["mrr_at_k"]
        }
    }

def main():
    ground_truth_file = "data/ground_truth.json"
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: {ground_truth_file} does not exist.")
        return

    # Configuration Grid Benchmarking
    exp_500 = run_grid_experiment(chunk_size=500, chunk_overlap=100, k=8, ground_truth_path=ground_truth_file)
    exp_1000 = run_grid_experiment(chunk_size=1000, chunk_overlap=150, k=8, ground_truth_path=ground_truth_file)

    benchmark_summary = {
        "experiment_500_chars": exp_500,
        "experiment_1000_chars": exp_1000
    }

    # Export grid benchmark results to results.json
    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n=======================================================")
    print(f"GRID BENCHMARK COMPLETE! Results exported to `{output_path}`")
    print("=======================================================")

if __name__ == "__main__":
    main()