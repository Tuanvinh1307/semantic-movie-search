"""
MODULE 11: FINAL SCORING & FILTERING
=========================================================
- Input: Danh sách phim đã qua Rerank (có ce_score và Fat Payload)
- Output: Top-N phim cuối cùng xuất ra API/UI.
- Logic: Kết hợp điểm AI (ce_score) với điểm TMDB (vote_average, popularity)
         để ưu tiên phim chất lượng cao.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class FinalScorer:
    def __init__(self, semantic_weight: float = 0.8, popularity_weight: float = 0.2):
        """
        Khởi tạo bộ tính điểm cuối.
        - semantic_weight: Trọng số cho điểm ngữ nghĩa của AI (Reranker).
        - popularity_weight: Trọng số cho độ hot/chất lượng của phim.
        """
        self.semantic_weight = semantic_weight
        self.popularity_weight = popularity_weight
        logger.info(f" Khởi tạo Final Scorer (AI: {semantic_weight * 100}% | Metadata: {popularity_weight * 100}%)")

    def normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Chuẩn hóa điểm số về thang 0.0 -> 1.0 (Min-Max Scaling)"""
        if max_val == min_val:
            return 0.0
        # Đảm bảo giá trị không vượt quá giới hạn
        value = max(min_val, min(value, max_val))
        return (value - min_val) / (max_val - min_val)

    def score_and_filter(self, reranked_movies: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        print("\n" + "=" * 80)
        print(" MODULE 11: FINAL SCORING & FILTERING")
        print("=" * 80)

        if not reranked_movies:
            return []

        # 1. TÌM MIN-MAX ĐỂ CHUẨN HÓA (Dựa trên tập kết quả hiện tại)
        ce_scores = [m.get("ce_score", 0.0) for m in reranked_movies]
        min_ce, max_ce = min(ce_scores), max(ce_scores)

        # 2. TÍNH ĐIỂM TỔNG HỢP (FINAL SCORE)
        for movie in reranked_movies:
            # Lấy điểm AI
            raw_ce = movie.get("ce_score", 0.0)
            norm_ce = self.normalize_score(raw_ce, min_ce, max_ce)

            # Lấy điểm Metadata (Vote Average thang 10)
            # Nếu phim không có vote, mặc định cho 5.0
            vote_avg = movie.get("vote_average", 5.0)
            norm_vote = self.normalize_score(vote_avg, 0.0, 10.0)

            # Công thức Mix: Semantic + Metadata
            final_score = (norm_ce * self.semantic_weight) + (norm_vote * self.popularity_weight)

            # Gán điểm mới và làm tròn
            movie["final_score"] = round(final_score, 4)

            # Dọn dẹp các trường nội bộ không cần thiết đưa ra UI
            movie.pop("max_score", None)
            movie.pop("matched_chunks", None)

        # 3. XẾP HẠNG LẦN CUỐI & CẮT NGỌN
        final_list = sorted(reranked_movies, key=lambda x: (x["final_score"], x.get("popularity", 0.0)), reverse=True)
        final_list = final_list[:top_n]

        logger.info(f" Đã xuất ra Top {len(final_list)} phim xuất sắc nhất.")

        # In log kết quả cuối cùng
        for i, m in enumerate(final_list[:3]):
            logger.info(
                f"    #{i + 1} | {m['title']} | Final: {m['final_score']} (AI: {m['ce_score']:.2f}, Vote: {m['vote_average']})")

        return final_list