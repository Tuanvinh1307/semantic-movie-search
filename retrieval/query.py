"""
MODULE 5: QUERY PROCESSING & ENCODING (ENGLISH)
========================================================================
- Input: User Query (raw string)
- Output: clean_query, dense_vector (384-dim), sparse_vector (BM25)
"""

import re
import logging
from typing import Tuple, List

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryEncoder:
    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        sparse_model_name: str = "Qdrant/bm25"
    ):
        logger.info(" Initializing Query Encoder...")
        logger.info(f"   - Dense: {dense_model_name}")
        logger.info(f"   - Sparse: {sparse_model_name}")

        self.dense_model = SentenceTransformer(dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

        logger.info(" Models ready!")

    # ============================================
    # STEP 1: CLEAN QUERY (ENGLISH)
    # ============================================
    def clean_query(self, query: str) -> str:
        if not isinstance(query, str):
            return ""

        # Remove HTML
        query = re.sub(r"<[^>]+>", " ", query)

        # Lowercase
        query = query.lower()

        # Remove punctuation (keep letters + numbers)
        query = re.sub(r"[^a-z0-9\s]", " ", query)

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query).strip()

        return query

    # ============================================
    # STEP 2 + 3: ENCODE
    # ============================================
    def encode(
        self, raw_query: str
    ) -> Tuple[str, List[float], Tuple[List[int], List[float]]]:

        # Step 1: Clean
        clean_q = self.clean_query(raw_query)

        if not clean_q:
            raise ValueError("Query is empty after cleaning!")

        logger.info(f" Query: '{clean_q}'")

        # Step 2: Dense
        dense_vec = self.dense_model.encode(
            clean_q,
            normalize_embeddings=True
        ).tolist()

        # Step 2: Sparse
        sparse_result = list(self.sparse_model.embed([clean_q]))[0]

        # Step 3: Format for Qdrant
        sparse_vec = (
            sparse_result.indices.tolist(),
            sparse_result.values.tolist()
        )

        return clean_q, dense_vec, sparse_vec