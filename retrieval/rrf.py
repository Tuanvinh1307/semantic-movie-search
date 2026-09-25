"""
MODULE 7: RANK FUSION (RRF)
========================================================================
- Input: dense_results, sparse_results (từ Module 6)
- Output: Danh sách chunks đã được trộn và xếp hạng lại
- Thuật toán: Reciprocal Rank Fusion (RRF)
- Tham số chuẩn: k = 60
- Feature:
    + Deduplicate
    + Merge payload
    + Top-K output
    + Fail-safe
"""

import logging
from typing import Dict, List, Any

# ============================================
# SETUP LOGGING
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RankFusion:
    def __init__(self, rrf_k: int = 60):
        """
        rrf_k = 60 là giá trị chuẩn (best practice)
        """
        self.rrf_k = rrf_k
        logger.info(f"Rank Fusion initialized (RRF k={self.rrf_k})")

    # ============================================
    # RRF SCORE
    # ============================================
    def rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    # ============================================
    # MAIN FUSION
    # ============================================
    def fuse(
        self,
        dense_results: Dict,
        sparse_results: Dict,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:

        logger.info("Starting Rank Fusion (RRF)...")

        # ===== FAIL-SAFE =====
        if not dense_results:
            dense_results = {}
        if not sparse_results:
            sparse_results = {}

        fused_map: Dict[str, Dict[str, Any]] = {}

        # =========================================================
        # 1. DENSE BRANCH
        # =========================================================
        dense_ids = dense_results.get("chunk_ids", [])
        dense_payloads = dense_results.get("payloads", [])

        for idx, chunk_id in enumerate(dense_ids):
            rank = idx + 1
            score = self.rrf_score(rank)

            fused_map[chunk_id] = {
                "chunk_id": chunk_id,
                "rrf_score": score,
                "payload": dense_payloads[idx] if idx < len(dense_payloads) else {},
                "in_dense": True,
                "in_sparse": False
            }

        # =========================================================
        # 2. SPARSE BRANCH
        # =========================================================
        sparse_ids = sparse_results.get("chunk_ids", [])
        sparse_payloads = sparse_results.get("payloads", [])

        for idx, chunk_id in enumerate(sparse_ids):
            rank = idx + 1
            score = self.rrf_score(rank)

            if chunk_id in fused_map:
                # cộng điểm nếu xuất hiện ở cả 2
                fused_map[chunk_id]["rrf_score"] += score
                fused_map[chunk_id]["in_sparse"] = True

                #  merge payload
                if idx < len(sparse_payloads):
                    fused_map[chunk_id]["payload"].update(sparse_payloads[idx])

            else:
                fused_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "rrf_score": score,
                    "payload": sparse_payloads[idx] if idx < len(sparse_payloads) else {},
                    "in_dense": False,
                    "in_sparse": True
                }

        # =========================================================
        # 3. SORTING
        # =========================================================
        fused_list = sorted(
            fused_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        logger.info(f"Fusion done | total unique chunks: {len(fused_list)}")

        # =========================================================
        # 4. TOP-K OUTPUT
        # =========================================================
        final_results = fused_list[:top_k]

        # Debug top 1
        if final_results:
            top_1 = final_results[0]
            logger.info(
                f"Top 1 → ID={top_1['chunk_id']} | "
                f"Score={top_1['rrf_score']:.6f} | "
                f"Dense={top_1['in_dense']} | Sparse={top_1['in_sparse']}"
            )

        return final_results