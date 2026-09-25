"""
MODULE 6: 1ST RETRIEVAL (HYBRID SEARCH)
========================================================================
- Input: dense_vector, sparse_vector
- Output: dense_results, sparse_results
- Feature: Hybrid Search + Metadata Filtering + Parallel Execution
"""

import os
import logging
import time
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

COLLECTION_NAME = "movies_hybrid_collection"

DENSE_K = 100
SPARSE_K = 100

# GLOBAL QDRANT CLIENT (CONNECTION POOLING)
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0
)

# ============================================
# DENSE RETRIEVAL
# ============================================
def dense_retrieval_cloud(
    dense_vector: List[float],
    k: int = DENSE_K,
    query_filter: Optional[models.Filter] = None
) -> Dict:

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

    return {
        "method": "dense",
        "chunk_ids": [str(hit.id) for hit in response.points],
        "scores": [hit.score for hit in response.points],
        "payloads": [hit.payload for hit in response.points],
        "count": len(response.points)
    }


# ============================================
# SPARSE RETRIEVAL
# ============================================
def sparse_retrieval_cloud(
    sparse_vector: Tuple[List[int], List[float]],
    k: int = SPARSE_K,
    query_filter: Optional[models.Filter] = None
) -> Dict:

    if not sparse_vector or not sparse_vector[0]:
        raise ValueError("Sparse vector is empty!")

    logger.info(f"Sparse Retrieval | Top-K={k}")
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
    logger.info(f"Sparse done in {elapsed:.4f}s")

    return {
        "method": "sparse",
        "chunk_ids": [str(hit.id) for hit in response.points],
        "scores": [hit.score for hit in response.points],
        "payloads": [hit.payload for hit in response.points],
        "count": len(response.points)
    }


# ============================================
# HYBRID RETRIEVAL (PARALLEL + SAFE)
# ============================================
def hybrid_retrieval(
    dense_vector: List[float],
    sparse_vector: Tuple[List[int], List[float]],
    dense_k: int = DENSE_K,
    sparse_k: int = SPARSE_K,
    query_filter: Optional[models.Filter] = None
) -> Tuple[Dict, Dict]:

    print("\n" + "=" * 80)
    print("MODULE 6: HYBRID RETRIEVAL (1st)")
    print("=" * 80)

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        future_dense = executor.submit(
            dense_retrieval_cloud,
            dense_vector,
            dense_k,
            query_filter
        )

        future_sparse = executor.submit(
            sparse_retrieval_cloud,
            sparse_vector,
            sparse_k,
            query_filter
        )

        try:
            dense_results = future_dense.result()
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            dense_results = {
                "method": "dense",
                "chunk_ids": [],
                "scores": [],
                "payloads": [],
                "count": 0
            }

        try:
            sparse_results = future_sparse.result()
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            sparse_results = {
                "method": "sparse",
                "chunk_ids": [],
                "scores": [],
                "payloads": [],
                "count": 0
            }

    total_time = time.time() - start_time
    logger.info(f"Hybrid retrieval done in {total_time:.4f}s")

    return dense_results, sparse_results