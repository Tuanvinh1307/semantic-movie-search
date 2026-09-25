"""
MODULE 1: DATA INGESTION (Thu thập dữ liệu từ TMDB)
=====================================================

Yêu cầu từ PDF:
- Input: TMDB API key từ .env
- Output: pipeline/data/movies_raw.csv
- Fields: movie_id, title, overview, genres, release_date, vote_average, popularity,
          director, cast, keywords, original_language, poster_path
- Thư viện: requests, pandas, python-dotenv
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import logging
from typing import Dict, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "movies_raw.csv")

# API endpoints
GENRE_LIST_URL = "https://api.themoviedb.org/3/genre/movie/list"
DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie"

# Rate limiting
REQUEST_DELAY = 0.05  # 50ms giữa các requests (tránh bị block API)


# ============================================
# HELPER FUNCTIONS
# ============================================

def fetch_genre_mapping() -> Dict[int, str]:
    """
    Lấy từ điển mapping: Genre ID (28) → Genre Name (Action)

    Returns:
        Dictionary: {genre_id: genre_name, ...}
        Example: {28: "Action", 12: "Adventure", ...}
    """
    logger.info(" Fetching genre mapping from TMDB...")

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    try:
        response = requests.get(GENRE_LIST_URL, params=params, timeout=10)

        if response.status_code == 200:
            genres = response.json().get("genres", [])
            genre_dict = {g["id"]: g["name"] for g in genres}
            logger.info(f"✓ Fetched {len(genre_dict)} genres")
            return genre_dict
        else:
            logger.error(f" Error fetching genres: HTTP {response.status_code}")
            return {}

    except Exception as e:
        logger.error(f" Connection error while fetching genres: {e}")
        return {}


def fetch_movie_details(movie_id: int) -> Dict:
    """
    Fetch chi tiết phim: credits (director, cast) + keywords

    Args:
        movie_id: TMDB movie ID

    Returns:
        Dictionary chứa:
        - director: Tên đạo diễn (hoặc empty string)
        - cast: List top 5 diễn viên (hoặc empty list)
        - keywords: List từ khóa (hoặc empty list)
    """

    details_url = f"{MOVIE_DETAILS_URL}/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,keywords"
    }

    try:
        response = requests.get(details_url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract director
            crew = data.get("credits", {}).get("crew", [])
            director = next(
                (member["name"] for member in crew if member["job"] == "Director"),
                ""
            )

            # Extract top 5 cast members
            cast_list = data.get("credits", {}).get("cast", [])
            top_cast = [actor["name"] for actor in cast_list[:5]]

            # Extract keywords
            kw_list = data.get("keywords", {}).get("keywords", [])
            keywords = [kw["name"] for kw in kw_list]

            return {
                "director": director,
                "cast": top_cast,
                "keywords": keywords
            }
        else:
            logger.warning(f"Error fetching details for movie {movie_id}: HTTP {response.status_code}")
            return {"director": "", "cast": [], "keywords": []}

    except Exception as e:
        logger.warning(f"Error fetching details for movie {movie_id}: {e}")
        return {"director": "", "cast": [], "keywords": []}

    finally:
        time.sleep(REQUEST_DELAY)


def fetch_tmdb_movies_by_year(
        start_year: int = 2023,
        end_year: int = 2024,
        max_pages_per_year: int = 10
) -> pd.DataFrame:
    """
    Thu thập metadata phim từ TMDB theo năm phát hành

    Args:
        start_year: Năm bắt đầu (inclusive)
        end_year: Năm kết thúc (inclusive)
        max_pages_per_year: Số trang tối đa để fetch mỗi năm

    Returns:
        DataFrame chứa tất cả phim với fields:
        - movie_id, title, overview, release_date
        - vote_average, popularity
        - genres (string, comma-separated)
        - director, cast (string, comma-separated), keywords (string, comma-separated)
        - original_language, poster_path

    Raises:
        ValueError: Nếu TMDB_API_KEY không được tìm thấy
    """

    if not TMDB_API_KEY:
        raise ValueError(" TMDB_API_KEY not found in .env file")

    logger.info("=" * 70)
    logger.info(f" MODULE 1: DATA INGESTION")
    logger.info(f" Fetching movies from {start_year} to {end_year}")
    logger.info("=" * 70)

    # Fetch genre mapping
    genre_mapping = fetch_genre_mapping()
    if not genre_mapping:
        logger.warning(" Genre mapping is empty, will proceed without genres")

    movies_data = []
    total_processed = 0

    # Loop qua từng năm
    for year in range(start_year, end_year + 1):
        logger.info(f"\n Processing year {year}...")
        movies_in_year = 0

        # Loop qua từng trang
        for page in range(1, max_pages_per_year + 1):
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page,
                "sort_by": "popularity.desc",
                "primary_release_year": year,
                "vote_count.gte": 100,
                "vote_average.gte": 6.5
            }

            try:
                response = requests.get(DISCOVER_URL, params=params, timeout=10)

                if response.status_code == 200:
                    results = response.json().get("results", [])

                    # Nếu không có kết quả, dừng pagination cho năm này
                    if not results:
                        logger.info(f"  No more movies for year {year}")
                        break

                    # Process mỗi phim trong trang này
                    for item in results:
                        movie_id = item.get("id")

                        # Fetch detail info (director, cast, keywords)
                        details = fetch_movie_details(movie_id)

                        # Convert genre IDs to names
                        genre_ids = item.get("genre_ids", [])
                        genre_names = [
                            genre_mapping.get(gid)
                            for gid in genre_ids
                            if gid in genre_mapping
                        ]

                        # Build movie record với tất cả required fields
                        movie_record = {
                            "movie_id": movie_id,
                            "title": item.get("title", ""),
                            "overview": item.get("overview", ""),
                            "release_date": item.get("release_date", ""),
                            "vote_average": item.get("vote_average", 0.0),
                            "popularity": item.get("popularity", 0.0),
                            "genres": ", ".join(genre_names) if genre_names else "",
                            "director": details.get("director", ""),
                            "cast": ", ".join(details.get("cast", [])) if details.get("cast") else "",
                            "keywords": ", ".join(details.get("keywords", [])) if details.get("keywords") else "",
                            "original_language": item.get("original_language", "en"),
                            "poster_path": item.get("poster_path", "")
                        }
                        movies_data.append(movie_record)
                        movies_in_year += 1

                    logger.info(f"  Page {page}: +{len(results)} movies (total: {len(movies_data)})")

                else:
                    logger.error(f" API error for year {year}, page {page}: HTTP {response.status_code}")
                    break

                # Rate limiting để tránh bị block API
                time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.error(f" Connection error on year {year}, page {page}: {e}")
                break

        logger.info(f"✓ Year {year} complete: +{movies_in_year} movies")
        total_processed += movies_in_year

    logger.info(f"\n Total movies fetched: {len(movies_data)}")

    # ============================================
    # DATA CLEANING
    # ============================================

    logger.info("\n Data cleaning...")

    df = pd.DataFrame(movies_data)

    if len(df) == 0:
        logger.error(" No movies fetched!")
        return df

    logger.info(f"  Before cleaning: {len(df)} movies")

    # Remove rows with empty overview
    df = df.dropna(subset=['overview'])
    df = df[df['overview'].str.strip() != '']
    logger.info(f"  After removing empty overview: {len(df)} movies")

    # Remove duplicates based on movie_id
    df = df.drop_duplicates(subset=['movie_id'])
    logger.info(f"  After removing duplicates: {len(df)} movies")

    # ============================================
    # SAVE TO CSV
    # ============================================

    # Create directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    logger.info("=" * 70)
    logger.info(f" MODULE 1 COMPLETE!")
    logger.info(f" Saved {len(df)} movies to: {OUTPUT_FILE}")
    logger.info(f" Columns: {', '.join(df.columns.tolist())}")
    logger.info(f" File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
    logger.info("=" * 70)

    return df


# ============================================
# MAIN
# ============================================

def main():
    """
    Entry point cho Module 1
    """

    try:
        df = fetch_tmdb_movies_by_year(
            start_year=1990,
            end_year=2026,
            max_pages_per_year=25
        )

        # Print first few rows để verify
        logger.info("\n First 3 rows of data:")
        logger.info(df.head(3).to_string())

    except Exception as e:
        logger.error(f" Error: {e}")
        raise


if __name__ == "__main__":
    main()