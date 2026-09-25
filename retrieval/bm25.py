"""
MODULE: BM25 ONLY RETRIEVAL
========================================================================
- Input: sparse_vector (BM25 embeddings)
- Output: ranked results (movie_id, score, metadata)
- Feature: Pure lexical search without dense vectors
"""

import os
import logging
import time
from typing import Tuple, Dict, Optional, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

COLLECTION_NAME = "movies_hybrid_collection"
SPARSE_K = 100

# GLOBAL QDRANT CLIENT
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0
)


# ============================================
# BM25 RETRIEVAL
# ============================================
def bm25_retrieval(
    sparse_vector: Tuple[List[int], List[float]],
    k: int = SPARSE_K,
    query_filter: Optional[models.Filter] = None
) -> Dict:
    """
    Pure BM25 (Sparse) Retrieval
    
    Args:
        sparse_vector: Tuple of (indices, values) from BM25 embedding
        k: Number of top results to retrieve
        query_filter: Optional Qdrant filter for metadata filtering
        
    Returns:
        Dict with retrieved movie chunks and metadata
    """
    
    print("\n" + "=" * 80)
    print("MODULE: BM25 RETRIEVAL (SPARSE ONLY)")
    print("=" * 80)

    if not sparse_vector or not sparse_vector[0]:
        raise ValueError("Sparse vector is empty!")

    logger.info(f"BM25 Retrieval | Top-K={k}")
    start_time = time.time()

    qdrant_sparse_vec = models.SparseVector(
        indices=sparse_vector[0],
        values=sparse_vector[1]
    )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        using="sparse",
        query=qdrant_sparse_vec,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
        with_vectors=False
    )

    elapsed = time.time() - start_time
    logger.info(f"BM25 done in {elapsed:.4f}s")

    results = {
        "method": "bm25",
        "chunk_ids": [str(hit.id) for hit in response.points],
        "scores": [hit.score for hit in response.points],
        "payloads": [hit.payload for hit in response.points],
        "count": len(response.points),
        "elapsed_time": elapsed
    }
    
    logger.info(f"Retrieved {results['count']} chunks")

    return results
