#  MovieScout AI: Hệ thống Tìm kiếm Phim Ngữ nghĩa (Advanced RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)
![Groq](https://img.shields.io/badge/Groq-Llama--3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)

**MovieScout AI** là một hệ thống tìm kiếm thông tin điện ảnh ứng dụng kiến trúc **Retrieval-Augmented Generation (RAG) Nâng cao**. Thay vì tìm kiếm theo từ khóa (Keyword matching) truyền thống, hệ thống cho phép người dùng tìm kiếm phim dựa trên **ngữ nghĩa, bối cảnh, hoặc những mảnh ký ức mơ hồ** (VD: *"phim khoa học viễn tưởng về không gian có cảnh bố để lại đồng hồ cho con gái"*).

Dự án này được phát triển nhằm giải quyết bài toán "bất đồng ranh giới từ vựng" (Vocabulary Mismatch) trong lĩnh vực Information Retrieval (IR), với khả năng xử lý hàng ngàn dữ liệu phim và trả về kết quả dưới 2 giây.

---

##  Tính năng nổi bật (Key Features)

Hệ thống được thiết kế theo kiến trúc **Adaptive Cascade Retrieval (Truy hồi Đa tầng Thích ứng)**, bao gồm các công nghệ lõi:

* ** Hybrid Search (Tìm kiếm Lai):** Kết hợp sức mạnh hiểu ngữ nghĩa của **Dense Vector** (`all-MiniLM-L6-v2`) và khả năng bắt từ khóa siêu việt của **Sparse Vector** (Thuật toán Okapi BM25 qua `fastembed`). Điểm số được dung hợp công bằng qua thuật toán **Reciprocal Rank Fusion (RRF)**.
* ** Difficulty Router (Định tuyến Thông minh):** Hệ thống tự động phân tích "khoảng cách điểm" (Score Gap) của kết quả vòng 1 để rẽ nhánh. Truy vấn dễ (EASY) sẽ trả kết quả ngay (Early Exit) để tối ưu tốc độ. Truy vấn khó (HARD) sẽ được gửi đi xử lý chuyên sâu.
* **🪄 Mở rộng Truy vấn với HyDE & Guardrail:** Sử dụng LLM **Llama-3.1-8b** (via Groq Cloud) để sinh "cốt truyện giả định" (HyDE) cho các truy vấn mập mờ. Đặc biệt, tích hợp **Module HyDE Validation** sử dụng Cross-Encoder (với Threshold -2.0) làm hàng rào chống ảo giác (Hallucination Guardrail), tự động ngắt mạch nếu LLM sinh nội dung lạc đề.
* ** Cross-Encoder Reranking:** Sử dụng mô hình AI đọc hiểu sâu `ms-marco-MiniLM-L-6-v2` để chấm điểm tương tác chéo (Cross-Attention) giữa câu hỏi và đoạn văn bản, mang lại độ chính xác xếp hạng tuyệt đối (MRR & NDCG cao).
* ** Qdrant Cloud Payload Indexing:** Áp dụng bộ lọc siêu dữ liệu (Pre-filtering) ngay trên cơ sở dữ liệu Vector để lọc phim theo Thể loại và Năm phát hành với độ trễ gần như bằng 0.

---

##  Cấu trúc hệ thống (System Architecture)

Hệ thống được chia làm hai Pipeline chính:

1.  **Offline Pipeline (Data Ingestion & Embedding):**
    * Tự động crawl dữ liệu chất lượng cao từ **TMDB API** (phim từ 1990-2026, vote > 6.5).
    * Tiền xử lý và phân mảnh văn bản (Recursive Chunking với `tiktoken`).
    * Mã hóa Vector (Batch Encoding) và đẩy dữ liệu lên **Qdrant Cloud** với cơ chế định danh UUID v5 chống trùng lặp.
2.  **Online Pipeline (Adaptive Retrieval):**
    * Tiếp nhận truy vấn từ giao diện **Streamlit**.
    * Thực hiện luồng xử lý: *Query Processing -> 1st Hybrid Search -> MaxP Aggregation -> Difficulty Routing -> HyDE (LLM) -> HyDE Validation -> 2nd Hybrid Search -> Cross-Encoder Rerank -> Final Scoring*.

---

##  Cài đặt & Khởi chạy (Installation & Setup)

### 1. Yêu cầu hệ thống
* Python 3.10 trở lên.
* Tài khoản Qdrant Cloud, Groq Cloud và TMDB (để lấy API Keys).

### 2. Cài đặt thư viện
Clone repository và cài đặt các dependencies:
```bash
git clone [https://github.com/your-username/semantic-movie-search.git](https://github.com/your-username/semantic-movie-search.git)
cd semantic-movie-search
pip install -r requirements.txt

### Cách chạy giao diện
streamlit run ui/app_final.py