"""
SQLite-based caching for evaluation results.

Provides persistent caching to avoid redundant LLM/VLM calls during evaluation.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


class BaseEvaluationCache:
    """
    Base class for SQLite-based evaluation caches.

    Provides common functionality for storing and retrieving cached evaluation scores.
    Uses SHA256 hashing for cache keys to enable fast lookups via primary key.
    """

    def __init__(self, cache_file: str, table_name: str):
        """
        Initialize cache with database file and table name.

        Args:
            cache_file: Path to SQLite database file
            table_name: Name of the cache table
        """
        self.cache_file = Path(cache_file)
        self.table_name = table_name
        self.db_url = f"sqlite:///{self.cache_file}"
        self._init_db()

    def _init_db(self):
        """Initialize database and create table. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _init_db")

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate unique cache key. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_cache_key")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        total_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT AVG(score) FROM {self.table_name}")
        avg_score = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT MIN(created_at), MAX(created_at)
            FROM {self.table_name}
        """)
        min_date, max_date = cursor.fetchone()

        conn.close()

        return {
            "total_entries": total_count,
            "average_score": avg_score,
            "oldest_entry": min_date,
            "newest_entry": max_date
        }

    def clear(self):
        """Clear all cached entries."""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name}")
        conn.commit()
        conn.close()

        print(f"Cleared all entries from {self.cache_file}")


class TextCorrectnessCache(BaseEvaluationCache):
    """
    Persistent SQLite-based cache for text-based LLM judge correctness scores.

    Uses SHA256 hash of (answer, ground_truth, prompt_hash) as cache key to avoid
    redundant LLM calls for identical evaluations.

    Attributes:
        cache_file: Path to SQLite database file
        prompt_hash: Hash of the evaluation prompt (to invalidate cache on prompt changes)
    """

    def __init__(self, cache_file: str = "answer_correctness_cache.db", prompt_hash: str = ""):
        """
        Initialize text correctness cache.

        Args:
            cache_file: Path to SQLite database file (default: "answer_correctness_cache.db")
            prompt_hash: Hash of the evaluation prompt for cache invalidation
        """
        self.prompt_hash = prompt_hash
        super().__init__(cache_file, "correctness_cache")

    def _init_db(self):
        """Initialize database and create table if it doesn't exist."""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        # Create table with hash as primary key for fast lookups
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correctness_cache (
                cache_key TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on created_at for potential cleanup operations
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON correctness_cache(created_at)
        """)

        conn.commit()

        # Get cache size
        cursor.execute("SELECT COUNT(*) FROM correctness_cache")
        count = cursor.fetchone()[0]

        conn.close()

        if count > 0:
            print(f"Loaded {count} cached correctness scores from {self.cache_file}")

    def _get_cache_key(self, answer: str, ground_truth: str) -> str:
        """Generate unique cache key from answer and ground truth."""
        combined = json.dumps({
            "answer": answer.strip(),
            "ground_truth": ground_truth.strip(),
            "prompt_hash": self.prompt_hash.strip()
        }, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, answer: str, ground_truth: str) -> float | None:
        """Get cached score if available."""
        cache_key = self._get_cache_key(answer, ground_truth)

        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT score FROM correctness_cache WHERE cache_key = ?",
            (cache_key,)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def set(self, answer: str, ground_truth: str, score: float):
        """Store score in cache."""
        cache_key = self._get_cache_key(answer, ground_truth)

        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        # Use INSERT OR REPLACE to handle both new and updated entries
        cursor.execute("""
            INSERT OR REPLACE INTO correctness_cache
            (cache_key, answer, ground_truth, score, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (cache_key, answer.strip(), ground_truth.strip(), score))

        conn.commit()
        conn.close()


class VlmCorrectnessCache(BaseEvaluationCache):
    """
    Persistent SQLite-based cache for VLM (vision-language model) correctness scores.

    Uses SHA256 hash of (answer, question, image_hashes, prompt_hash) as cache key
    to avoid redundant VLM calls for identical evaluations with same images.

    Attributes:
        cache_file: Path to SQLite database file
        prompt_hash: Hash of the evaluation prompt (to invalidate cache on prompt changes)
    """

    def __init__(self, cache_file: str = "vlm_answer_correctness_cache.db", prompt_hash: str = ""):
        """
        Initialize VLM correctness cache.

        Args:
            cache_file: Path to SQLite database file (default: "vlm_answer_correctness_cache.db")
            prompt_hash: Hash of the evaluation prompt for cache invalidation
        """
        self.prompt_hash = prompt_hash
        super().__init__(cache_file, "vlm_correctness_cache")

    def _init_db(self):
        """Initialize database and create table if it doesn't exist."""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        # Create table with hash as primary key for fast lookups
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vlm_correctness_cache (
                cache_key TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                question TEXT NOT NULL,
                image_hashes TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on created_at for potential cleanup operations
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vlm_created_at
            ON vlm_correctness_cache(created_at)
        """)

        conn.commit()

        # Get cache size
        cursor.execute("SELECT COUNT(*) FROM vlm_correctness_cache")
        count = cursor.fetchone()[0]

        conn.close()

        if count > 0:
            print(f"Loaded {count} cached VLM correctness scores from {self.cache_file}")

    def _get_cache_key(self, answer: str, question: str, image_filenames: list[str]) -> str:
        """Generate unique cache key from answer, question, and image filenames."""
        # Sort image filenames for consistent hashing
        sorted_images = sorted(image_filenames)
        combined = json.dumps({
            "answer": answer.strip(),
            "question": question.strip(),
            "images": sorted_images,
            "prompt_hash": self.prompt_hash
        }, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, answer: str, question: str, image_filenames: list[str]) -> float | None:
        """Get cached score if available."""
        cache_key = self._get_cache_key(answer, question, image_filenames)

        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT score FROM vlm_correctness_cache WHERE cache_key = ?",
            (cache_key,)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def set(self, answer: str, question: str, image_filenames: list[str], score: float):
        """Store score in cache."""
        cache_key = self._get_cache_key(answer, question, image_filenames)
        image_hashes = ",".join(sorted(image_filenames))

        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        # Use INSERT OR REPLACE to handle both new and updated entries
        cursor.execute("""
            INSERT OR REPLACE INTO vlm_correctness_cache
            (cache_key, answer, question, image_hashes, score, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (cache_key, answer.strip(), question.strip(), image_hashes, score))

        conn.commit()
        conn.close()
