"""
MODULE 8: CHUNK AGGREGATION (MAXP - PRO VERSION)
========================================================================
- Input: Top-K chunks từ Rank Fusion
- Output: Top-N movies với metadata đầy đủ
- Thuật toán:
    + MaxP (Maximum Passage Pooling)
    + Logarithmic Boost (configurable)
- Improvements:
    + O(1) duplicate check (set)
    + Memory optimized (no full payload)
    + Overflow protection (top-k chunks/movie)
    + Clean scoring pipeline
"""

import logging
import math
from typing import Dict, List, Any

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkAggregator:
    def __init__(
        self,
        top_n_movies: int = 10,
        max_chunks_per_movie: int = 5,
        boost_alpha: float = 0.05
    ):
        self.top_n_movies = top_n_movies
        self.max_chunks_per_movie = max_chunks_per_movie
        self.boost_alpha = boost_alpha

        logger.info(
            f" ChunkAggregator initialized | Top-{top_n_movies} | "
            f"MaxChunks={max_chunks_per_movie} | BoostAlpha={boost_alpha}"
        )

    # =========================================================
    # MAIN AGGREGATION FUNCTION
    # =========================================================
    def aggregate(self, ranked_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f" Aggregating {len(ranked_chunks)} chunks...")

        movie_map: Dict[int, Dict[str, Any]] = {}

        for chunk in ranked_chunks:
            payload = chunk.get("payload", {})
            movie_id = payload.get("movie_id")

            if movie_id is None:
                logger.warning(f"️ Skip chunk {chunk.get('chunk_id')} (missing movie_id)")
                continue

            score = chunk.get("rrf_score", 0.0)

            chunk_info = {
                "chunk_id": chunk.get("chunk_id"),
                "score": score,
                "text": payload.get("chunk_text", "")
            }

            # =================================================
            # INIT MOVIE
            # =================================================
            if movie_id not in movie_map:
                movie_map[movie_id] = {
                    # Metadata (Fat Payload)
                    "movie_id": movie_id,
                    "title": payload.get("title", "Unknown"),
                    "genres": payload.get("genres", ""),
                    "release_date": payload.get("release_date", ""),
                    "vote_average": payload.get("vote_average", 0.0),
                    "poster_path": payload.get("poster_path", ""),

                    # Scoring
                    "max_score": score,

                    # Evidence tracking
                    "best_chunk": chunk_info,
                    "matched_chunks": [chunk_info],
                    "chunk_ids": {chunk_info["chunk_id"]},  # O(1) duplicate check
                    "total_chunks_matched": 1
                }
                continue

            movie = movie_map[movie_id]

            # =================================================
            # DUPLICATE CHECK (O(1))
            # =================================================
            if chunk_info["chunk_id"] in movie["chunk_ids"]:
                continue

            movie["chunk_ids"].add(chunk_info["chunk_id"])

            # =================================================
            # MAXP UPDATE
            # =================================================
            if score > movie["max_score"]:
                movie["max_score"] = score
                movie["best_chunk"] = chunk_info

                # push best chunk lên đầu
                movie["matched_chunks"].insert(0, chunk_info)

            else:
                if len(movie["matched_chunks"]) < self.max_chunks_per_movie:
                    movie["matched_chunks"].append(chunk_info)

            # overflow protection
            if len(movie["matched_chunks"]) > self.max_chunks_per_movie:
                movie["matched_chunks"].pop()

            movie["total_chunks_matched"] += 1

        # =========================================================
        # FINAL SCORING
        # =========================================================
        ranked_movies = []

        for movie in movie_map.values():
            n = movie["total_chunks_matched"]

            # Log boost
            boost = self.boost_alpha * math.log(n + 1)

            final_score = movie["max_score"] + boost

            movie["movie_score"] = final_score

            # Remove internal fields (clean output)
            movie.pop("chunk_ids", None)

            ranked_movies.append(movie)

        # =========================================================
        # SORTING
        # =========================================================
        ranked_movies.sort(
            key=lambda x: (x["movie_score"], x["max_score"]),
            reverse=True
        )

        logger.info(
            f" Done: {len(ranked_chunks)} chunks → {len(ranked_movies)} movies"
        )

        # Top-N
        final_movies = ranked_movies[:self.top_n_movies]

        if final_movies:
            top = final_movies[0]
            logger.info(
                f" Top 1: {top['title']} | Score={top['movie_score']:.6f} | "
                f"Chunks={top['total_chunks_matched']}"
            )

        return final_movies