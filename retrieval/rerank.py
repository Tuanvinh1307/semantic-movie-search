"""
MODULE 10: CROSS-ENCODER RERANKING
=========================================================
- Input: Câu hỏi người dùng (query) & Danh sách Top 10 phim (List of Dicts) từ Mod 8/9.
- Output: Danh sách phim đã được xếp hạng lại với điểm `ce_score`.
- Lợi ích: Khai thác toàn bộ mảng `matched_chunks` làm Full Context.
"""

import logging
import time
import torch
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f" Khởi tạo Cross-Encoder Reranker trên {self.device.upper()}...")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Thiếu thư viện sentence-transformers!")

        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"    Đã nạp thành công model: {model_name}")
        except Exception as e:
            logger.error(f" Lỗi load Cross-Encoder model: {e}")
            raise

    def rerank(self, query: str, top_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("\n" + "=" * 80)
        print(" MODULE 10: CROSS-ENCODER RERANKING")
        print("=" * 80)

        if not top_movies:
            logger.warning(" Không có bộ phim nào để Rerank.")
            return []

        logger.info(f" Đang trích xuất Full Docs của {len(top_movies)} bộ phim...")

        # CHUẨN BỊ FULL DOCS (Gộp toàn bộ matched_chunks)
        pairs = []
        for movie in top_movies:
            title = movie.get("title", "Unknown")
            genres = movie.get("genres", "N/A")

            matched_chunks = movie.get("matched_chunks", [])
            evidence_texts = [chunk.get("text", "") for chunk in matched_chunks]
            combined_evidence = " ".join(evidence_texts)

            doc_text = f"Title: {title}. Genres: {genres}. Context: {combined_evidence}"
            pairs.append([query, doc_text])

        logger.info(f" Đang so khớp {len(pairs)} cặp câu bằng Cross-Attention...")
        start_time = time.time()

        try:
            scores = self.model.predict(pairs)
            for i, movie in enumerate(top_movies):
                movie["ce_score"] = float(scores[i])

            logger.info(f"    Chấm điểm xong trong {time.time() - start_time:.3f}s")

        except Exception as e:
            logger.error(f" Lỗi Inference CE: {e}")
            for movie in top_movies:
                movie["ce_score"] = movie.get("movie_score", 0.0)

        # SẮP XẾP LẠI
        reranked_movies = sorted(top_movies, key=lambda x: x["ce_score"], reverse=True)

        if reranked_movies:
            logger.info(" KẾT QUẢ RERANK (TOP 3):")
            for idx, movie in enumerate(reranked_movies[:3]):
                logger.info(f"   #{idx + 1} | {movie['title']} | CE Score: {movie['ce_score']:.4f}")

        return reranked_movies