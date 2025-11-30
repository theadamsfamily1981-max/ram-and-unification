"""Avatar generation caching system.

Caches generated avatar videos to avoid redundant processing.
Uses content-based hashing for cache keys.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AvatarCache:
    """Avatar generation cache manager."""

    def __init__(
        self,
        cache_dir: Path = Path("cache/avatars"),
        max_size_mb: int = 5000,
        ttl_hours: int = 24,
        compress: bool = True
    ):
        """Initialize avatar cache.

        Args:
            cache_dir: Cache directory
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
            compress: Enable compression for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours
        self.compress = compress

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        # Load metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    self._metadata = json.load(f)
                logger.info(f"Loaded cache metadata: {len(self._metadata)} entries")
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def generate_key(
        self,
        image_path: Path,
        audio_path: Path,
        **kwargs
    ) -> str:
        """Generate cache key from inputs using chunked reads.

        Args:
            image_path: Path to avatar image
            audio_path: Path to audio file
            **kwargs: Additional parameters that affect generation

        Returns:
            Cache key (SHA256 hash)
        """
        hasher = hashlib.sha256()

        # CHUNK_SIZE to avoid blocking on large files
        CHUNK_SIZE = 64 * 1024  # 64KB chunks

        # Hash image file in chunks to avoid blocking on large files
        try:
            with open(image_path, 'rb') as f:
                while chunk := f.read(CHUNK_SIZE):
                    hasher.update(chunk)
        except (IOError, OSError) as e:
            logger.error(f"Error reading image file {image_path}: {e}")
            # Fallback: hash the path string instead
            hasher.update(str(image_path).encode())

        # Hash audio file in chunks
        try:
            with open(audio_path, 'rb') as f:
                while chunk := f.read(CHUNK_SIZE):
                    hasher.update(chunk)
        except (IOError, OSError) as e:
            logger.error(f"Error reading audio file {audio_path}: {e}")
            # Fallback: hash the path string instead
            hasher.update(str(audio_path).encode())

        # Hash additional parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        hasher.update(params_str.encode())

        return hasher.hexdigest()

    async def generate_key_async(
        self,
        image_path: Path,
        audio_path: Path,
        **kwargs
    ) -> str:
        """Async version of generate_key for use in async contexts.

        Args:
            image_path: Path to avatar image
            audio_path: Path to audio file
            **kwargs: Additional parameters that affect generation

        Returns:
            Cache key (SHA256 hash)
        """
        import asyncio
        import concurrent.futures

        # Run blocking I/O in thread pool executor
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.generate_key(image_path, audio_path, **kwargs)
            )

    def get(self, cache_key: str) -> Optional[Path]:
        """Get cached video.

        Args:
            cache_key: Cache key

        Returns:
            Path to cached video or None
        """
        if cache_key not in self._metadata:
            return None

        entry = self._metadata[cache_key]
        cache_path = Path(entry["path"])

        # Check if file exists
        if not cache_path.exists():
            logger.warning(f"Cache entry missing file: {cache_key}")
            del self._metadata[cache_key]
            self._save_metadata()
            return None

        # Check TTL
        age_hours = (time.time() - entry["created_at"]) / 3600
        if age_hours > self.ttl_hours:
            logger.info(f"Cache entry expired: {cache_key} (age: {age_hours:.1f}h)")
            self._delete_entry(cache_key)
            return None

        # Update access time
        entry["last_accessed"] = time.time()
        entry["access_count"] = entry.get("access_count", 0) + 1
        self._save_metadata()

        logger.info(f"Cache hit: {cache_key}")
        return cache_path

    def put(
        self,
        cache_key: str,
        video_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Put video in cache.

        Args:
            cache_key: Cache key
            video_path: Path to video file
            metadata: Additional metadata

        Returns:
            Path to cached video
        """
        # Check cache size and cleanup if needed
        self._ensure_space()

        # Copy file to cache
        cache_file = self.cache_dir / f"{cache_key}.mp4"
        shutil.copy(video_path, cache_file)

        # Store metadata
        file_size = cache_file.stat().st_size
        self._metadata[cache_key] = {
            "path": str(cache_file),
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 1,
            "size_bytes": file_size,
            "metadata": metadata or {}
        }
        self._save_metadata()

        logger.info(f"Cached video: {cache_key} ({file_size / 1024 / 1024:.2f} MB)")
        return cache_file

    def _delete_entry(self, cache_key: str):
        """Delete cache entry.

        Args:
            cache_key: Cache key to delete
        """
        if cache_key not in self._metadata:
            return

        entry = self._metadata[cache_key]
        cache_path = Path(entry["path"])

        # Delete file
        if cache_path.exists():
            cache_path.unlink()

        # Delete metadata
        del self._metadata[cache_key]
        self._save_metadata()

        logger.info(f"Deleted cache entry: {cache_key}")

    def _ensure_space(self):
        """Ensure cache has enough space by deleting old entries."""
        # Calculate current cache size
        total_size = sum(
            entry["size_bytes"]
            for entry in self._metadata.values()
        )
        total_size_mb = total_size / 1024 / 1024

        if total_size_mb < self.max_size_mb:
            return

        logger.info(f"Cache full ({total_size_mb:.1f} MB), cleaning up...")

        # Sort by last access time (oldest first)
        entries = sorted(
            self._metadata.items(),
            key=lambda x: x[1]["last_accessed"]
        )

        # Delete oldest entries until under limit
        target_size_mb = self.max_size_mb * 0.8  # Leave 20% headroom
        for cache_key, entry in entries:
            self._delete_entry(cache_key)
            total_size_mb -= entry["size_bytes"] / 1024 / 1024

            if total_size_mb < target_size_mb:
                break

        logger.info(f"Cache cleanup complete: {total_size_mb:.1f} MB")

    def clear(self):
        """Clear entire cache."""
        logger.info("Clearing cache...")

        # Delete all files
        for entry in self._metadata.values():
            cache_path = Path(entry["path"])
            if cache_path.exists():
                cache_path.unlink()

        # Clear metadata
        self._metadata = {}
        self._save_metadata()

        logger.info("Cache cleared")

    def cleanup_expired(self):
        """Remove expired cache entries."""
        logger.info("Cleaning up expired cache entries...")

        expired = []
        current_time = time.time()

        for cache_key, entry in self._metadata.items():
            age_hours = (current_time - entry["created_at"]) / 3600
            if age_hours > self.ttl_hours:
                expired.append(cache_key)

        for cache_key in expired:
            self._delete_entry(cache_key)

        logger.info(f"Removed {len(expired)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache stats dictionary
        """
        total_size = sum(
            entry["size_bytes"]
            for entry in self._metadata.values()
        )
        total_accesses = sum(
            entry.get("access_count", 0)
            for entry in self._metadata.values()
        )

        return {
            "total_entries": len(self._metadata),
            "total_size_mb": total_size / 1024 / 1024,
            "max_size_mb": self.max_size_mb,
            "utilization_percent": (total_size / 1024 / 1024) / self.max_size_mb * 100,
            "total_accesses": total_accesses,
            "ttl_hours": self.ttl_hours
        }
