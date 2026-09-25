"""
MODULE 4: DUAL EMBEDDING & VECTOR DB UPSERT (NO CHUNK MODE)
========================================================================
- Data source: movies_clean.csv (mỗi phim = 1 document)
- Hybrid Embedding: Dense + Sparse
- Deterministic UUID5 (no duplication)
- Batch processing + retry (An toàn tuyệt đối)
- Clean & safe data handling (Giữ nguyên vẹn token ngữ nghĩa)
"""

import os
import time
import pandas as pd
import logging
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient, models
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f" Import Error: {e}")

# ============================================
# CONFIG
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_MOVIES_CLEAN = os.path.join(BASE_DIR, "data", "movies_clean.csv")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "movies_hybrid_collection"

BATCH_SIZE = 32
MAX_RETRY = 3 # Tăng lên 3 lần cho chắc cốp với mạng quốc tế

# ============================================
# HELPER
# ============================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return "empty"
    text = text.strip().lower()
    return text if text else "empty"

# ============================================
# MAIN PROCESS
# ============================================

def process_dual_embedding():
    print("\n" + "=" * 100)
    print(" MODULE 4: DUAL EMBEDDING & UPSERT (PRODUCTION)")
    print("=" * 100)

    # 0. Validate
    if not DEPENDENCIES_OK:
        raise ImportError("Missing required libraries")

    if not os.path.exists(INPUT_MOVIES_CLEAN):
        raise FileNotFoundError(f"Missing file: {INPUT_MOVIES_CLEAN}")

    # 1. Init
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # 2. Load data (NO CHUNK MODE: one movie = one searchable document)
    df = pd.read_csv(INPUT_MOVIES_CLEAN)

    if "combined_text" not in df.columns:
        raise ValueError("movies_clean.csv is missing required column: combined_text")

    df = df.fillna({
        "title": "",
        "genres": "",
        "release_date": "",
        "poster_path": "",
        "vote_average": 0.0,
        "popularity": 0.0,
        "combined_text": ""
    })

    # Keep only rows with usable text
    df = df[df["combined_text"].astype(str).str.strip() != ""].copy()
    df["chunk_position"] = 0
    df["chunk_text"] = df["combined_text"]

    logger.info(f" Ready {len(df)} movie-documents (chunk module removed)")

    # 3. Collection
    dense_dim = dense_model.get_sentence_embedding_dimension()

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )

        # Index
        qdrant.create_payload_index(COLLECTION_NAME, "movie_id", models.PayloadSchemaType.INTEGER)
        qdrant.create_payload_index(COLLECTION_NAME, "title", models.PayloadSchemaType.TEXT)
        qdrant.create_payload_index(COLLECTION_NAME, "genres", models.PayloadSchemaType.TEXT)
        qdrant.create_payload_index(COLLECTION_NAME, "release_date", models.PayloadSchemaType.TEXT)
        logger.info(" Đã tạo Collection và đánh Index thành công!")
    else:
        logger.info(" Collection đã tồn tại. Sẵn sàng cập nhật/bổ sung dữ liệu.")

    # 4. Batch
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Upserting"):
        batch_df = df.iloc[i:i + BATCH_SIZE]

        # Clean text
        batch_texts = [clean_text(t) for t in batch_df['chunk_text'].tolist()]

        # Dense
        dense_vecs = dense_model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        ).tolist()

        # Sparse
        sparse_vecs = list(sparse_model.embed(batch_texts))

        points = []

        for j, (_, row) in enumerate(batch_df.iterrows()):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"movie_{row['movie_id']}"
            ))

            payload = {
                "movie_id": int(row['movie_id']),
                "chunk_position": int(row['chunk_position']),
                "chunk_text": str(row['chunk_text']),

                "title": str(row.get("title", "")),
                "genres": str(row.get("genres", "")),
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(row.get("vote_average", 0)),
                "popularity": float(row.get("popularity", 0)),
                "poster_path": str(row.get("poster_path", ""))
            }

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vecs[j],
                    "sparse": models.SparseVector(
                        indices=sparse_vecs[j].indices.tolist(),
                        values=sparse_vecs[j].values.tolist()
                    )
                },
                payload=payload
            ))

        for attempt in range(MAX_RETRY):
            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                logger.error(f" Upsert failed (attempt {attempt+1}/{MAX_RETRY}): {e}")
                time.sleep(2)
                if attempt == MAX_RETRY - 1:
                    raise Exception(f" MẤT MẠNG HOÀN TOÀN! Dừng hệ thống để bảo toàn dữ liệu. Lỗi: {e}")

    print("\n DONE: Data is ready in Qdrant!")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    process_dual_embedding()