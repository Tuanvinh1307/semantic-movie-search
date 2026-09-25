"""
MODULE: DENSE EMBEDDING RETRIEVAL
========================================================================
- Input: dense_vector (semantic embeddings)
- Output: ranked results (movie_id, score, metadata)
- Feature: Pure semantic search without lexical matching
"""

import os
import logging
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

COLLECTION_NAME = "movies_hybrid_collection"
DENSE_K = 100

# GLOBAL QDRANT CLIENT
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0
)


# ============================================
# DENSE RETRIEVAL
# ============================================
def dense_retrieval(
    dense_vector: List[float],
    k: int = DENSE_K,
    query_filter: Optional[models.Filter] = None
) -> Dict:
    """
    Pure Dense (Semantic) Retrieval
    
    Args:
        dense_vector: Dense embedding vector (typically 384-dim)
        k: Number of top results to retrieve
        query_filter: Optional Qdrant filter for metadata filtering
        
    Returns:
        Dict with retrieved movie chunks and metadata
    """
    
    print("\n" + "=" * 80)
    print("MODULE: DENSE RETRIEVAL (SEMANTIC ONLY)")
    print("=" * 80)

    if not dense_vector:
        raise ValueError("Dense vector is empty!")

    logger.info(f"Dense Retrieval | Top-K={k}")
    start_time = time.time()

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        using="dense",
        query=dense_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
        with_vectors=False
    )

    elapsed = time.time() - start_time
    logger.info(f"Dense done in {elapsed:.4f}s")

    results = {
        "method": "dense",
        "chunk_ids": [str(hit.id) for hit in response.points],
        "scores": [hit.score for hit in response.points],
        "payloads": [hit.payload for hit in response.points],
        "count": len(response.points),
        "elapsed_time": elapsed
    }
    
    logger.info(f"Retrieved {results['count']} chunks")

    return results
