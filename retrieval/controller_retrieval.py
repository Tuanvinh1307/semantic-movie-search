"""
ONLINE RETRIEVAL PIPELINE
==============================================================
1st Retrieval → Router → (Early Exit | HyDE → 2nd → Rerank) → Final

- Fix confidence_index logic (Dùng max_score thuần túy)
- Router thông minh (Gap + Absolute Score)
- Tích hợp Qdrant Pre-filtering (Thể loại, Năm) trực tiếp vào Hybrid Search
- Đã sửa lệnh return để tương thích với Giao diện Streamlit mới (trả về 3 giá trị)
"""

import time
import logging
import os
from typing import List, Dict, Any, Tuple

from query import QueryEncoder
from hybrid import hybrid_retrieval
from rrf import RankFusion
from aggregate import ChunkAggregator
from hyde import HyDEProcessor
from rerank import CrossEncoderReranker
from final_scorer import FinalScorer
from qdrant_client import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveSearchPipeline:
    def __init__(self, ci_threshold: float = 0.01, min_score_threshold: float = 0.03):
        logger.info(" Initializing Adaptive Search Pipeline...")
        start_load = time.time()

        self.query_encoder = QueryEncoder()
        self.rank_fusion = RankFusion(rrf_k=60)

        self.aggregator_1st = ChunkAggregator(max_chunks_per_movie=5)
        self.aggregator_2nd = ChunkAggregator(max_chunks_per_movie=10)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY environment variable")

        self.hyde = HyDEProcessor(api_key=groq_api_key)

        self.reranker = CrossEncoderReranker()
        self.final_scorer = FinalScorer(semantic_weight=0.8, popularity_weight=0.2)

        self.ci_threshold = ci_threshold
        self.min_score_threshold = min_score_threshold

        logger.info(f" System ready! Load time: {time.time() - start_load:.2f}s")

    def _build_qdrant_filter(self, user_filters: dict):
        if not user_filters:
            return None

        must_conditions = []

        # 1. Lọc Thể loại
        genre = user_filters.get('genre')
        if genre and genre != "Tất cả":
            must_conditions.append(
                models.FieldCondition(key="genres", match=models.MatchText(text=genre))
            )

        # 2. Lọc Khoảng năm
        year_input = user_filters.get('year', '').strip()
        if year_input:
            # Trường hợp 1: Nhập khoảng (VD: 2000-2020 hoặc 2000 to 2020)
            if "-" in year_input or " to " in year_input.lower():
                separator = "-" if "-" in year_input else " to "
                parts = year_input.lower().split(separator)
                if len(parts) == 2:
                    start_year = parts[0].strip()
                    end_year = parts[1].strip()
                    must_conditions.append(
                        models.FieldCondition(
                            key="release_date",
                            range=models.Range(
                                gte=f"{start_year}-01-01",
                                lte=f"{end_year}-12-31"
                            )
                        )
                    )
            # Trường hợp 2: Nhập năm đơn lẻ (VD: 2014)
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="release_date",
                        match=models.MatchText(text=year_input)
                    )
                )

        if must_conditions:
            return models.Filter(must=must_conditions)
        return None

    def search(self, raw_query: str, top_n: int = 10, user_filters: dict = None) -> Tuple[
        List[Dict[str, Any]], str, str]:
        total_start = time.time()
        logger.info(f"\n{'=' * 70}\n QUERY: {raw_query}\n{'=' * 70}")

        # KHỞI TẠO 2 BIẾN NÀY ĐỂ CHUẨN BỊ RETURN CHO STREAMLIT
        route_name = "EASY"
        hypo_doc_text = None

        # --- KHỞI TẠO BỘ LỌC TỪ UI ---
        qdrant_filter = self._build_qdrant_filter(user_filters)

        # =========================================================
        # 1ST RETRIEVAL
        # =========================================================
        clean_q, orig_dense, orig_sparse = self.query_encoder.encode(raw_query)

        # TRUYỀN BỘ LỌC VÀO VÒNG 1
        dense_res_1, sparse_res_1 = hybrid_retrieval(
            dense_vector=orig_dense,
            sparse_vector=orig_sparse,
            query_filter=qdrant_filter
        )

        fused_1 = self.rank_fusion.fuse(dense_res_1, sparse_res_1)
        movies_1st = self.aggregator_1st.aggregate(fused_1)

        # =========================================================
        # DIFFICULTY ROUTER (IMPROVED)
        # =========================================================
        if len(movies_1st) > 1:
            top1 = movies_1st[0]["max_score"]
            top2 = movies_1st[1]["max_score"]
            gap = top1 - top2

            logger.info(f" Gap = {top1:.4f} - {top2:.4f} = {gap:.4f}")

            if top1 < self.min_score_threshold:
                confidence_index = 0.0
                logger.info("️ Top1 score quá thấp → ép HARD")
            else:
                confidence_index = gap

        elif len(movies_1st) == 1:
            confidence_index = 0.0
            logger.info("️ Chỉ có 1 kết quả → ép HARD")

        else:
            confidence_index = 0.0
            logger.info(" Không có kết quả → HARD")

        # =========================================================
        # ROUTING
        # =========================================================
        if confidence_index >= self.ci_threshold:
            logger.info(f" EASY (Gap={confidence_index:.4f}) → EARLY EXIT")

            route_name = "EASY"  # Ghi nhận luồng EASY

            candidates = movies_1st
            for m in candidates:
                m["ce_score"] = m["movie_score"]

        else:
            logger.info(f" HARD (Gap={confidence_index:.4f}) → HYDE PIPELINE")

            route_name = "HARD"  # Ghi nhận luồng HARD

            # =====================================================
            # HYDE
            # =====================================================
            hyde_dense, hypo_doc, _ = self.hyde.get_hyde_vector(clean_q)

            hypo_doc_text = hypo_doc  # Ghi nhận cốt truyện giả định để ném ra Streamlit

            # TRUYỀN BỘ LỌC VÀO VÒNG 2
            dense_res_2, sparse_res_2 = hybrid_retrieval(
                dense_vector=hyde_dense,
                sparse_vector=orig_sparse,
                query_filter=qdrant_filter
            )

            fused_2 = self.rank_fusion.fuse(dense_res_2, sparse_res_2)
            movies_2nd = self.aggregator_2nd.aggregate(fused_2)

            # =====================================================
            # RERANK (dynamic top-k)
            # =====================================================
            rerank_k = 10 if confidence_index > 0.005 else 20
            logger.info(f" Rerank Top {rerank_k}")

            candidates = self.reranker.rerank(clean_q, movies_2nd[:rerank_k])

        # =========================================================
        # FINAL SCORING
        # =========================================================
        final_results = self.final_scorer.score_and_filter(candidates, top_n=top_n)

        logger.info(f" Done in {time.time() - total_start:.3f}s")

        return final_results, route_name, hypo_doc_text