"""
MODULE 15: STREAMLIT WEB APP (GIAO DIỆN NGƯỜI DÙNG CLOUD-NATIVE)
===============================================================
- Giao diện Ultra Premium kết nối với AdaptiveSearchPipeline.
- Có bộ lọc Pre-filtering (Thể loại, Năm) giao tiếp với Qdrant.
- Hiển thị linh hoạt luồng định tuyến (EASY/HARD) và cốt truyện HyDE.
"""

import streamlit as st
import time
import os
import sys

# 1. Lấy đường dẫn của thư mục 'ui' hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Lùi lại 1 cấp để ra thư mục gốc
project_root = os.path.dirname(current_dir)

# 3. Trỏ thẳng vào thư mục 'retrieval'
retrieval_dir = os.path.join(project_root, "retrieval")

# 4. Ép Python phải cho cả 2 thư mục này vào tầm nhìn
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if retrieval_dir not in sys.path:
    sys.path.insert(0, retrieval_dir)

# 5. BÂY GIỜ THÌ IMPORT CHẮC CHẮN SẼ THÀNH CÔNG
from retrieval.controller_retrieval import AdaptiveSearchPipeline

# ============================================
# CONFIG TRANG
# ============================================
st.set_page_config(
    page_title="MovieScout AI - Semantic Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CSS TÙY CHỈNH (ULTRA PREMIUM UI)
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@300;400;500;600&display=swap');

html, body {
    background:#0B0F1A;
    font-family:'Inter',sans-serif;
}

/* Nền tảng với hiệu ứng ánh sáng gradient */
.stApp {
    background: radial-gradient(circle at 30% 20%, rgba(124,58,237,0.15), transparent 40%),
                radial-gradient(circle at 80% 70%, rgba(34,211,238,0.1), transparent 40%),
                #0B0F1A;
}

/* HEADER */
.header { text-align:center; margin-bottom:30px; }
.header h1 {
    font-family:'Orbitron';
    font-size:3.5rem;
    background:linear-gradient(135deg,#7C3AED,#22D3EE);
    -webkit-background-clip:text;
    color:transparent;
    margin-bottom: 5px;
}
.header p { color:#94A3B8; font-size: 1.1rem;}

/* CARD PHIM */
.movie-card {
    background:rgba(17,24,39,0.85);
    border-radius:15px;
    padding:20px;
    border:1px solid rgba(124,58,237,0.3);
    margin-bottom:20px;
    transition:0.3s;
}
.movie-card:hover {
    transform:translateY(-5px);
    box-shadow:0 0 25px rgba(124,58,237,0.4);
}

.movie-title {
    font-size:1.6rem;
    font-weight:700;
    color:#E5E7EB;
    margin-bottom: 8px;
}

.badge {
    padding:6px 14px;
    border-radius:20px;
    font-size:14px;
    font-weight: bold;
}
.score {background:#22D3EE;color:#0B0F1A;}
.rating {background:#7C3AED;color:white; margin-left: 8px;}

.genre {
    border:1px solid #7C3AED;
    border-radius:15px;
    padding:4px 12px;
    font-size:13px;
    display:inline-block;
    margin: 10px 5px 15px 0;
    color: #E0E0E0;
}

.movie-info { color: #A0A0A0; font-size: 14.5px; line-height: 1.6;}
.movie-info b { color: #22D3EE; }

/* FOOTER */
.footer { text-align:center; margin-top:50px; color:#6B7280; }
</style>
""", unsafe_allow_html=True)


# ============================================
# HÀM BÓC TÁCH DỮ LIỆU TỪ "FAT PAYLOAD"
# ============================================
def parse_movie_info(raw_text):
    """Trích xuất Đạo diễn, Diễn viên, Tóm tắt từ khối text kết hợp của Module 2"""
    director, cast, plot = "Updating", "Updating", raw_text
    try:
        if "Director:" in raw_text and ". Cast:" in raw_text:
            director = raw_text.split("Director:")[1].split(". Cast:")[0].strip()
        if "Cast:" in raw_text and ". Genres:" in raw_text:
            cast = raw_text.split("Cast:")[1].split(". Genres:")[0].strip()
        if "Overview:" in raw_text:
            plot = raw_text.split("Overview:")[1].strip()
    except Exception:
        pass
    return director, cast, plot


# ============================================
# NẠP HỆ THỐNG AI (CHẠY 1 LẦN DUY NHẤT)
# ============================================
@st.cache_resource(show_spinner=" The system is starting up...")
def init_engine():
    # Sử dụng ngưỡng CI tối ưu mà ông giáo đã chốt
    return AdaptiveSearchPipeline(ci_threshold=0.01)


engine = init_engine()

# ============================================
# GIAO DIỆN CHÍNH
# ============================================
st.markdown("""
<div class='header'>
<h1>MovieScout AI</h1>
<p>Tell me the plot, and I’ll find the movie title for you!</p>
</div>
""", unsafe_allow_html=True)

# --- KHU VỰC BỘ LỌC (Lọc trước khi tính toán) ---
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    genres_list = ["Tất cả", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family",
                   "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War",
                   "Western"]
    selected_genre = st.selectbox("🎭 Lọc theo thể loại", genres_list)

with col_filter2:
    selected_year = st.text_input("📅 Lọc theo năm phát hành", placeholder="Ví dụ: 2014 (Để trống nếu không rõ)")

st.markdown("<br>", unsafe_allow_html=True)

# --- THANH TÌM KIẾM ---
query = st.text_input(" Bạn đang nhớ mang máng bộ phim nào?",
                      placeholder="Ví dụ: a guy gets trapped inside a video game...")

col_btn1, col_btn2, col_btn3 = st.columns([4, 2, 4])
with col_btn2:
    search_button = st.button(" Bắt đầu quét dữ liệu", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# XỬ LÝ TÌM KIẾM
# ============================================
if query or search_button:
    if not query.strip():
        st.warning("Vui lòng nhập một vài từ khóa để bắt đầu tìm kiếm!")
    else:
        # 1. Đóng gói Bộ lọc từ UI
        user_filters = {}
        if selected_genre != "Tất cả":
            user_filters['genre'] = selected_genre
        if selected_year.strip():
            user_filters['year'] = selected_year.strip()

        with st.spinner("Hệ thống đang tìm phim phù hợp với bạn, bạn chờ chút nhé..."):
            start_time = time.time()

            # 2. GỌI PIPELINE
            results, route_name, hyde_plot = engine.search(query, top_n=10, user_filters=user_filters)

            latency = round(time.time() - start_time, 2)

        if results:
            # BẢNG THỐNG KÊ NHANH
            st.success("Tìm kiếm hoàn tất!")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("⏱️ Thời gian phản hồi", f"{latency}s")
            col_s2.metric("🎯 Số lượng phim tìm thấy", f"{len(results)} phim")

            # Cập nhật hiển thị luồng (EASY/HARD)
            if route_name == "HARD":
                col_s3.metric("⚙️ Cơ chế hoạt động", "HARD (HyDE + Rerank)")
            else:
                col_s3.metric("⚙️ Cơ chế hoạt động", "EASY (Early Exit)")

            st.divider()

            # HIỂN THỊ CỐT TRUYỆN GIẢ ĐỊNH NẾU LÀ NHÁNH HARD VÀ CÓ DỮ LIỆU
            if route_name == "HARD" and hyde_plot:
                st.info(f"** Cốt truyện giả định (HyDE) do AI Llama-3 sinh ra để mở rộng truy vấn:**\n\n*{hyde_plot}*")
                st.divider()

            # RENDER TỪNG BỘ PHIM
            for idx, movie in enumerate(results):

                # --- Xử lý hình ảnh ---
                p_path = movie.get('poster_path', '')
                if p_path and str(p_path).lower() not in ['none', 'null', '']:
                    full_poster_url = f"https://image.tmdb.org/t/p/w500{p_path}"
                else:
                    full_poster_url = "https://via.placeholder.com/500x750/111827/7C3AED?text=No+Poster"

                # --- Xử lý Text từ Fat Payload ---
                best_chunk_text = movie.get('best_chunk', {}).get('text', '')
                director, cast, plot = parse_movie_info(best_chunk_text)

                # --- Xử lý Metadata ---
                year = str(movie.get('release_date', 'N/A'))[:4]
                genres_list_movie = str(movie.get('genres', '')).split(',')
                genre_html = "".join(
                    [f"<span class='genre'>{g.strip()}</span>" for g in genres_list_movie if g.strip()])

                final_score = movie.get('final_score', 0.0)
                vote_avg = movie.get('vote_average', 0.0)

                # --- Vẽ giao diện ---
                col_img, col_txt = st.columns([1.5, 4.5])

                with col_img:
                    st.image(full_poster_url, use_container_width=True)

                with col_txt:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">#{idx + 1} {movie.get('title', 'Unknown')} ({year})</div>
                        <div style="margin: 12px 0;">
                            <span class="badge score">🔥 Độ khớp: {final_score:.4f}</span>
                            <span class="badge rating">⭐ {vote_avg}/10 IMDb</span>
                        </div>
                        <div>{genre_html}</div>
                        <div class="movie-info">
                            <p><b>🎬 Đạo diễn:</b> {director}</p>
                            <p><b>🎭 Diễn viên:</b> {cast}</p>
                            <p><b>📖 Nội dung:</b> {plot}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error(
                "Xin lỗi, hệ thống không có phim nào khớp với miêu tả và bộ lọc của bạn!")

st.markdown("<div class='footer'>Đừng coi phim nữa, học bài đi!</div>",
            unsafe_allow_html=True)