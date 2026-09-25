"""
MODULE 2: DOCUMENT PROCESSING (Tiền xử lý văn bản)
===================================================

Yêu cầu:
- Input: pipeline/data/movies_raw.csv
- Output: pipeline/data/movies_clean.csv (có thêm combined_text)
- Xử lý:
  1. Làm sạch văn bản: loại bỏ HTML tags, URLs, chuẩn hóa whitespace
  2. Giữ nguyên stopwords (không xóa từ dừng)
  3. Tạo combined_text: ghép title + genres + director + cast + keywords + overview

combined_text ví dụ:
"Title: Interstellar. Director: Christopher Nolan. Cast: Matthew McConaughey, Anne Hathaway. 
Genres: Adventure, Drama, Science Fiction. Keywords: wormhole, black hole, time travel, space... 
Overview: The adventures of a group of explorers..."
"""

import pandas as pd
import re
import os
import logging

# ============================================
# SETUP LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

# Lấy đường dẫn tuyệt đối của thư mục chứa file clean.py (thư mục 'pipeline')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "movies_raw.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "movies_clean.csv")

# Text fields to clean
TEXT_FIELDS = ['title', 'overview', 'director', 'cast', 'keywords', 'genres']

# Fields to include in combined_text
COMBINED_FIELDS = {
    'title': 'Title',
    'director': 'Director',
    'cast': 'Cast',
    'genres': 'Genres',
    'keywords': 'Keywords',
    'overview': 'Overview'
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_text(text):
    """
    Làm sạch định dạng văn bản:
    - Remove HTML tags
    - Remove URLs
    - Remove extra whitespace
    - Giữ nguyên stopwords

    Args:
        text: Input text (string or NaN)

    Returns:
        Cleaned text string
    """

    # Handle non-string and NaN values
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Remove HTML tags (e.g., <p>, <div>, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs (http://, https://, www., etc.)
    text = re.sub(r'http\S+|www\.\S+', '', text, flags=re.MULTILINE)

    # Remove extra whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def create_combined_text(row):
    """
    Tạo combined_text bằng cách ghép các fields:
    Title, Director, Cast, Genres, Keywords, Overview

    Format:
    "Title: XXX. Director: YYY. Cast: ZZZ. Genres: AAA. Keywords: BBB. Overview: CCC."

    Args:
        row: Pandas Series (một dòng dữ liệu)

    Returns:
        combined_text string
    """

    parts = []

    # Add Title
    if row['title']:
        parts.append(f"Title: {row['title']}")

    # Add Director
    if row['director']:
        parts.append(f"Director: {row['director']}")

    # Add Cast
    if row['cast']:
        parts.append(f"Cast: {row['cast']}")

    # Add Genres
    if row['genres']:
        parts.append(f"Genres: {row['genres']}")

    # Add Keywords
    if row['keywords']:
        parts.append(f"Keywords: {row['keywords']}")

    # Add Overview
    if row['overview']:
        parts.append(f"Overview: {row['overview']}")

    # Join all parts with ". "
    combined = ". ".join(parts)

    return combined


def process_documents():
    """
    Main pipeline:
    1. Load raw data
    2. Fill NaN values
    3. Clean text fields
    4. Create combined_text
    5. Remove empty rows
    6. Save cleaned data
    """

    print("\n" + "=" * 100)
    print(" MODULE 2: DOCUMENT PROCESSING")
    print("=" * 100)

    # ============================================
    # 1. Load data
    # ============================================

    if not os.path.exists(INPUT_FILE):
        logger.error(f" Input file not found: {INPUT_FILE}")
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    logger.info(f" Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    before_count = len(df)
    logger.info(f"   ✓ Loaded {before_count} movies")

    # ============================================
    # 2. Fill NaN with empty strings
    # ============================================

    logger.info("\n Filling NaN values...")
    nan_before = df.isnull().sum().sum()
    df = df.fillna('')
    nan_after = df.isnull().sum().sum()
    logger.info(f"   ✓ Filled {nan_before} NaN values")

    # ============================================
    # 3. Clean text fields (remove HTML, URLs, extra spaces)
    # ============================================

    logger.info("\n Cleaning text fields...")
    for field in TEXT_FIELDS:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
            logger.info(f"   ✓ Cleaned {field}")

    # ============================================
    # 4. Create combined_text field (CRITICAL!)
    # ============================================

    logger.info("\n Creating combined_text field...")
    logger.info("   Format: Title. Director. Cast. Genres. Keywords. Overview.")

    df['combined_text'] = df.apply(create_combined_text, axis=1)

    logger.info(f"   ✓ Created combined_text for {len(df)} movies")

    # Show example of combined_text
    if len(df) > 0:
        example_text = df.iloc[0]['combined_text']
        logger.info(f"\n   Example combined_text (first 200 chars):")
        logger.info(f"   {example_text[:200]}...")

    # ============================================
    # 5. Remove rows with empty combined_text
    # ============================================

    logger.info("\n  Removing rows with empty combined_text...")
    df_before_empty = len(df)
    df = df[df['combined_text'].str.strip() != '']
    empty_removed = df_before_empty - len(df)
    logger.info(f"   ✓ Removed {empty_removed} rows")

    # ============================================
    # 6. Remove rows with empty overview (backup check)
    # ============================================

    logger.info("\n  Removing rows with empty overview...")
    df_before_overview = len(df)
    df = df[df['overview'].str.strip() != '']
    overview_removed = df_before_overview - len(df)
    logger.info(f"   ✓ Removed {overview_removed} rows")

    # ============================================
    # 7. Remove duplicate movies (by movie_id)
    # ============================================

    logger.info("\n Removing duplicate movies...")
    df_before_dup = len(df)
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    duplicates_removed = df_before_dup - len(df)
    logger.info(f"   ✓ Removed {duplicates_removed} duplicates")

    # ============================================
    # 8. Verify combined_text was created
    # ============================================

    if 'combined_text' not in df.columns:
        raise ValueError(" combined_text field not created!")

    # Check no empty combined_text
    empty_combined = (df['combined_text'].str.strip() == '').sum()
    if empty_combined > 0:
        logger.warning(f"  Found {empty_combined} empty combined_text rows")

    # ============================================
    # 9. Save cleaned data
    # ============================================

    logger.info(f"\n Saving cleaned data...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    logger.info(f"   ✓ Saved to {OUTPUT_FILE}")

    # ============================================
    # 10. Print summary
    # ============================================

    print("\n" + "=" * 100)
    print(" MODULE 2 COMPLETE")
    print("=" * 100)
    print(f"\n SUMMARY:")
    print(f"   Movies before: {before_count}")
    print(f"   Movies after: {len(df)}")
    print(f"   Total removed: {before_count - len(df)}")
    print(f"     - Empty combined_text: {empty_removed}")
    print(f"     - Empty overview: {overview_removed}")
    print(f"     - Duplicates: {duplicates_removed}")

    print(f"\n Output columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"     - {col}")

    print(f"\n KEY FIELD: combined_text")
    print(f"   - Total combined_text created: {len(df)}")
    print(f"   - Avg combined_text length: {df['combined_text'].str.len().mean():.0f} chars")
    print(f"   - Min combined_text length: {df['combined_text'].str.len().min()} chars")
    print(f"   - Max combined_text length: {df['combined_text'].str.len().max()} chars")

    print(f"\n Output file: {OUTPUT_FILE}")
    print(f"   - File size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"   - Encoding: UTF-8")

    # Show sample cleaned data
    print(f"\n SAMPLE DATA (First movie):")
    if len(df) > 0:
        first_movie = df.iloc[0]
        print(f"   movie_id: {first_movie['movie_id']}")
        print(f"   title: {first_movie['title']}")
        print(f"   overview: {first_movie['overview'][:80]}...")
        print(f"   director: {first_movie['director']}")
        print(f"   genres: {first_movie['genres']}")
        print(f"   vote_average: {first_movie['vote_average']}")
        print(f"\n   combined_text (first 300 chars):")
        print(f"   {first_movie['combined_text'][:300]}...")

    print("=" * 100 + "\n")

    return df


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    try:
        df_clean = process_documents()
        print("Module 2 executed successfully!")
        print(f"OUTPUT: {OUTPUT_FILE} is ready for Module 3 (Chunking)")
    except Exception as e:
        logger.error(f" Error during document processing: {e}")
        raise