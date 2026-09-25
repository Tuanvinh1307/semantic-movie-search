"""
EVALUATION SCRIPT: BM25 vs Dense Retrieval
========================================================================
- Experiments: Compare pure BM25 and pure Dense on 200 evaluation queries
- Metrics: Hit@K, MRR@10
- Output: Detailed CSV + summary report
"""

import os
import sys
import time
import logging
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from query import QueryEncoder
from bm25 import bm25_retrieval
from dense import dense_retrieval
from aggregate import ChunkAggregator

# SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRetriever:
    """Simple retriever for evaluation (no HyDE, no reranking)"""
    
    def __init__(self, top_n: int = 10):
        logger.info(" Initializing Simple Retriever...")
        
        self.query_encoder = QueryEncoder()
        self.aggregator = ChunkAggregator(top_n_movies=top_n, max_chunks_per_movie=5)
        self.top_n = top_n
        
        logger.info(f" Retriever ready (Top-{top_n})")
    
    def retrieve_bm25(self, raw_query: str) -> List[Dict]:
        """Retrieve using BM25 only"""
        try:
            _, dense_vec, sparse_vec = self.query_encoder.encode(raw_query)
            
            # BM25 retrieval
            bm25_results = bm25_retrieval(sparse_vec, k=100)
            
            # Aggregate chunks to movie level
            chunks = []
            for chunk_id, score, payload in zip(
                bm25_results["chunk_ids"],
                bm25_results["scores"],
                bm25_results["payloads"]
            ):
                chunks.append({
                    "chunk_id": chunk_id,
                    "rrf_score": score,
                    "payload": payload
                })
            
            aggregated = self.aggregator.aggregate(chunks)
            return aggregated[:self.top_n]
        
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []
    
    def retrieve_dense(self, raw_query: str) -> List[Dict]:
        """Retrieve using Dense only"""
        try:
            _, dense_vec, sparse_vec = self.query_encoder.encode(raw_query)
            
            # Dense retrieval
            dense_results = dense_retrieval(dense_vec, k=100)
            
            # Aggregate chunks to movie level
            chunks = []
            for chunk_id, score, payload in zip(
                dense_results["chunk_ids"],
                dense_results["scores"],
                dense_results["payloads"]
            ):
                chunks.append({
                    "chunk_id": chunk_id,
                    "rrf_score": score,
                    "payload": payload
                })
            
            aggregated = self.aggregator.aggregate(chunks)
            return aggregated[:self.top_n]
        
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []


def calculate_metrics(
    results: List[Dict],
    expected_movie_id: int,
    top_k: int = 10
) -> Dict[str, bool]:
    """Calculate if expected movie is in top-K results"""
    
    retrieved_ids = [r.get("movie_id") for r in results[:top_k]]
    
    metrics = {
        "hit@1": expected_movie_id in retrieved_ids[:1],
        "hit@3": expected_movie_id in retrieved_ids[:3],
        "hit@5": expected_movie_id in retrieved_ids[:5],
        "hit@10": expected_movie_id in retrieved_ids[:top_k],
        "mrr": 0.0
    }
    
    # Calculate MRR
    if expected_movie_id in retrieved_ids:
        rank = retrieved_ids.index(expected_movie_id) + 1
        metrics["mrr"] = 1.0 / rank
    
    return metrics


def evaluate_batch(
    eval_file: str,
    output_dir: str = "evaluation",
    sample_size: int = None
) -> None:
    """
    Evaluate BM25 and Dense on evaluation queries
    
    Args:
        eval_file: Path to CSV with evaluation queries
        output_dir: Directory to save results
        sample_size: If set, only evaluate first N queries (for testing)
    """
    
    print("\n" + "=" * 100)
    print("EVALUATION: BM25 vs DENSE RETRIEVAL")
    print("=" * 100)
    
    # Load evaluation data
    logger.info(f"Loading evaluation queries from {eval_file}...")
    df = pd.read_csv(eval_file)
    
    if sample_size:
        df = df.head(sample_size)
        logger.info(f"Using first {sample_size} queries for quick test")
    
    logger.info(f"Total queries: {len(df)}")
    
    # Initialize retriever
    retriever = SimpleRetriever(top_n=10)
    
    # Results storage
    results_data = []
    bm25_metrics_agg = defaultdict(float)
    dense_metrics_agg = defaultdict(float)
    query_type_metrics = defaultdict(lambda: {"bm25": defaultdict(float), "dense": defaultdict(float)})
    
    # Evaluate each query
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    for idx, row in df.iterrows():
        query_id = row["query_id"]
        query_text = row["query"]
        query_type = row["query_type"]
        expected_movie_id = int(row["expected_movie_id"])
        expected_title = row["expected_title"]
        
        if (idx + 1) % 20 == 0:
            logger.info(f"Progress: {idx + 1}/{len(df)} queries processed...")
        
        # BM25 Retrieval
        bm25_results = retriever.retrieve_bm25(query_text)
        bm25_metrics = calculate_metrics(bm25_results, expected_movie_id)
        
        # Dense Retrieval
        dense_results = retriever.retrieve_dense(query_text)
        dense_metrics = calculate_metrics(dense_results, expected_movie_id)
        
        # Get top-1 results for logging
        bm25_top1 = bm25_results[0]["title"] if bm25_results else "N/A"
        dense_top1 = dense_results[0]["title"] if dense_results else "N/A"
        
        # Store results
        results_data.append({
            "query_id": query_id,
            "query": query_text,
            "query_type": query_type,
            "expected_movie_id": expected_movie_id,
            "expected_title": expected_title,
            "bm25_hit@1": int(bm25_metrics["hit@1"]),
            "bm25_hit@3": int(bm25_metrics["hit@3"]),
            "bm25_hit@5": int(bm25_metrics["hit@5"]),
            "bm25_hit@10": int(bm25_metrics["hit@10"]),
            "bm25_mrr": round(bm25_metrics["mrr"], 4),
            "bm25_top1": bm25_top1,
            "dense_hit@1": int(dense_metrics["hit@1"]),
            "dense_hit@3": int(dense_metrics["hit@3"]),
            "dense_hit@5": int(dense_metrics["hit@5"]),
            "dense_hit@10": int(dense_metrics["hit@10"]),
            "dense_mrr": round(dense_metrics["mrr"], 4),
            "dense_top1": dense_top1,
        })
        
        # Aggregate metrics
        for metric in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
            bm25_metrics_agg[metric] += bm25_metrics[metric]
            dense_metrics_agg[metric] += dense_metrics[metric]
            query_type_metrics[query_type]["bm25"][metric] += bm25_metrics[metric]
            query_type_metrics[query_type]["dense"][metric] += dense_metrics[metric]
    
    # Calculate final metrics
    n_queries = len(df)
    bm25_final = {k: v / n_queries for k, v in bm25_metrics_agg.items()}
    dense_final = {k: v / n_queries for k, v in dense_metrics_agg.items()}
    
    elapsed = time.time() - start_time
    avg_latency = elapsed / n_queries
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "comparison_results.csv")
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(results_csv, index=False)
    logger.info(f"✓ Detailed results saved to {results_csv}")
    
    # Generate summary report
    print("\n" + "=" * 100)
    print("SUMMARY: BM25 vs DENSE")
    print("=" * 100)
    
    print(f"\n📊 OVERALL METRICS (N={n_queries} queries)")
    print("-" * 100)
    print(f"{'Metric':<15} {'BM25':<20} {'Dense':<20} {'Winner':<15}")
    print("-" * 100)
    
    for metric in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
        bm25_val = bm25_final[metric]
        dense_val = dense_final[metric]
        winner = "BM25 ✓" if bm25_val > dense_val else ("Dense ✓" if dense_val > bm25_val else "Tie")
        print(f"{metric:<15} {bm25_val:<20.4f} {dense_val:<20.4f} {winner:<15}")
    
    print(f"\n⏱️  LATENCY")
    print("-" * 100)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per query: {avg_latency:.4f}s")
    
    # Per-query-type breakdown
    print(f"\n🔍 BREAKDOWN BY QUERY TYPE")
    print("-" * 100)
    
    for qtype in sorted(query_type_metrics.keys()):
        type_count = len(df[df["query_type"] == qtype])
        print(f"\n{qtype.upper()} ({type_count} queries)")
        print(f"  {'Metric':<15} {'BM25':<20} {'Dense':<20}")
        print(f"  {'-'*55}")
        
        for metric in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
            bm25_val = query_type_metrics[qtype]["bm25"][metric] / type_count
            dense_val = query_type_metrics[qtype]["dense"][metric] / type_count
            print(f"  {metric:<15} {bm25_val:<20.4f} {dense_val:<20.4f}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("EVALUATION SUMMARY: BM25 vs DENSE\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total Queries: {n_queries}\n")
        f.write(f"Total Time: {elapsed:.2f}s\n")
        f.write(f"Avg Latency: {avg_latency:.4f}s\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 100 + "\n")
        for metric in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
            f.write(f"{metric:<15} BM25: {bm25_final[metric]:<10.4f} Dense: {dense_final[metric]:<10.4f}\n")
        
        f.write("\nPER-QUERY-TYPE METRICS\n")
        for qtype in sorted(query_type_metrics.keys()):
            type_count = len(df[df["query_type"] == qtype])
            f.write(f"\n{qtype.upper()} ({type_count} queries)\n")
            for metric in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
                bm25_val = query_type_metrics[qtype]["bm25"][metric] / type_count
                dense_val = query_type_metrics[qtype]["dense"][metric] / type_count
                f.write(f"  {metric:<15} BM25: {bm25_val:<10.4f} Dense: {dense_val:<10.4f}\n")
    
    logger.info(f"✓ Summary report saved to {summary_file}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   - Detailed results: {results_csv}")
    print(f"   - Summary report: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BM25 vs Dense retrieval")
    parser.add_argument(
        "--eval_file",
        type=str,
        default="../evaluation/eval_queries_200.csv",
        help="Path to evaluation queries CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, only evaluate first N queries (for quick testing)"
    )
    
    args = parser.parse_args()
    
    evaluate_batch(
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        sample_size=args.sample
    )
