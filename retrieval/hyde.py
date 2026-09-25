"""
MODULE 9: QUERY EXPANSION (HyDE) - PURE GENERATION
==============================================================
- Chức năng: Chỉ thực thi sinh cốt truyện ảo khi được Router gọi.
- Groq API (Llama-3-8B): Tốc độ < 1s, siêu tốc.
- Caching: Lưu kết quả LLM để tiết kiệm API (Tối ưu I/O).
- Output: Dense Vector của cốt truyện ảo (Phục vụ 2nd Retrieval).
"""

import time
import logging
import re
from typing import Tuple, List

from sentence_transformers import SentenceTransformer
from groq import Groq

logger = logging.getLogger(__name__)

class HyDEProcessor:
    def __init__(self, api_key: str = None, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        # =========================
        # NẠP MÔ HÌNH NHÚNG (EMBEDDING)
        # =========================
        logger.info(f" Đang nạp mô hình Embedding cho HyDE: {embedding_model_name}")
        self.encoder = SentenceTransformer(embedding_model_name)

        # =========================
        # NẠP MÔ HÌNH TẠO SINH (GROQ LLAMA-3)
        # =========================
        logger.info("⚡ Đang nạp Groq API Client (Llama-3)...")
        self.api_key = api_key

        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.model_name = 'llama-3.1-8b-instant'
                logger.info("   ✓ Groq Client khởi tạo thành công.")
            except Exception as e:
                logger.error(f"    Lỗi khởi tạo Groq Client: {e}")
                self.client = None
        else:
            self.client = None
            logger.warning("   ️ Không có API Key. Cảnh báo: Sẽ không sinh được văn bản ảo!")

        # Bộ nhớ đệm (Tránh gọi API nhiều lần cho cùng 1 câu hỏi)
        self.cache = {}

    # ======================================================
    # LÕI SINH VĂN BẢN ẢO (LLM GENERATION)
    # ======================================================
    def _generate_hypo_doc(self, query: str) -> str:
        # 1. Kiểm tra Cache
        if query in self.cache:
            logger.info("    Bắt trúng HyDE Cache! Bỏ qua việc gọi API.")
            return self.cache[query]

        if not self.client:
            return query

        # 2. Xây dựng Prompt
        system_prompt = "You are an expert movie screenwriter. Output ONLY the raw plot premise. No titles, no intro, no fluff."
        user_prompt = f"Write a fast-paced, 3-sentence plot summary for a hypothetical movie that perfectly matches this request: '{query}'"

        try:
            start_gen = time.time()
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.7,
                max_tokens=150,
            )

            # 3. Làm sạch output của LLM
            hypo = chat_completion.choices[0].message.content.strip()
            hypo = re.sub(r"\s+", " ", hypo).strip()

            gen_time = time.time() - start_gen
            logger.info(f"    Llama-3 sinh cốt truyện ảo trong {gen_time:.3f}s")

            # Fallback nếu LLM bị ngáo (trả về quá ngắn)
            if len(hypo) < 20:
                hypo = query

            # Lưu vào Cache
            self.cache[query] = hypo
            return hypo

        except Exception as e:
            logger.error(f" Lỗi khi sinh văn bản bằng Groq: {e}")
            return query

    # ======================================================
    # HÀM GIAO TIẾP CHÍNH (ĐƯỢC GỌI BỞI PIPELINE)
    # ======================================================
    def get_hyde_vector(self, query: str) -> Tuple[List[float], str, float]:
        """
        Nhận lệnh từ Difficulty Router, sinh cốt truyện và trả về Vector Đặc.
        Trả về: (dense_vector, hypo_doc_text, latency)
        """
        start_time = time.time()

        logger.info(f" KÍCH HOẠT HyDE: Đang ảo hóa cốt truyện cho '{query}'...")

        # Gọi trực tiếp bộ sinh văn bản ảo (Vì Router đã chốt đây là câu Khó)
        doc = self._generate_hypo_doc(query)
        logger.debug(f"   -> Cốt truyện ảo: {doc}")

        # Chuẩn hóa chữ thường
        doc = doc.lower()

        # Nhúng thành Vector Đặc (Dense Vector) và chuẩn hóa (L2 Norm)
        vec = self.encoder.encode(
            doc,
            normalize_embeddings=True
        ).tolist()

        latency = round(time.time() - start_time, 4)
        return vec, doc, latency