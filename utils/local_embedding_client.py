import os
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.colorLogger import print_info, print_error, print_warning
from dotenv import load_dotenv
import hashlib
import pickle
import traceback
from pathlib import Path
import time
from tqdm import tqdm

load_dotenv()

# Enhanced model options with different capabilities and dimensions
EMBEDDING_MODELS = {
    "base": "all-MiniLM-L6-v2",  # 384 dimensions, fast
    "medium": "all-mpnet-base-v2",  # 768 dimensions, higher quality
    "large": "all-distilroberta-v1",  # 768 dimensions, excellent quality
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # Supports 50+ languages
}

# Default model - using a sentence transformers model with good balance of speed/quality
DEFAULT_LOCAL_EMBEDDING_MODEL = EMBEDDING_MODELS["base"]

# Target dimension for compatibility with OpenAI embeddings
OPENAI_DIMENSION = 1536


class LocalEmbeddingClient:
    """
    Enhanced client for handling embeddings using local Sentence Transformers models
    with dimension resizing, caching, and batch processing optimizations
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        normalize_embeddings: bool = True,
        target_dimension: int = OPENAI_DIMENSION,
    ):
        """
        Initialize the enhanced local embedding client
        Args:
            model_name: The name of the embedding model to use
            cache_dir: Directory to cache embeddings (defaults to ./.embedding_cache)
            use_cache: Whether to use embedding caching
            normalize_embeddings: Whether to normalize embeddings
            target_dimension: Target dimension for embeddings
        """
        try:
            # Use GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize the model with advanced options
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=os.getenv("MODELS_CACHE_DIR", "./.model_cache"),
            )

            # Get model's original dimension
            self.original_dim = self.model.get_sentence_embedding_dimension()

            # Store configuration
            self.model_name = model_name
            self.target_dim = target_dimension
            self.normalize = normalize_embeddings
            self.use_cache = use_cache

            # Setup caching if enabled
            if use_cache:
                self.cache_dir = cache_dir or os.getenv(
                    "EMBEDDING_CACHE_DIR", "./.embedding_cache"
                )
                os.makedirs(self.cache_dir, exist_ok=True)
                print_info(f"Embedding cache enabled at {self.cache_dir}")

                # Track cache stats
                self.cache_hits = 0
                self.cache_misses = 0

            print_info(
                f"Initialized enhanced embedding client with model: {model_name} on {self.device}"
                f" (Original dim: {self.original_dim}, Target dim: {self.target_dim})"
            )
        except Exception as e:
            print_error(f"Error initializing local embedding client: {str(e)}")
            print_error(f"Detailed Error Traceback: {traceback.format_exc()}")
            raise

    def _get_cache_path(self, text: str) -> str:
        """
        Get cache file path for a given text
        Args:
            text: Input text to generate cache key for
        Returns:
            Path to the cache file
        """
        # Create a hash of the text and model name for the cache key
        cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _resize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Resize embedding to target dimension using improved methods
        Args:
            embedding: The original embedding vector
        Returns:
            Resized embedding vector
        """
        if self.original_dim == self.target_dim:
            return embedding

        elif self.original_dim > self.target_dim:
            # Downsizing: use PCA-like approach (taking most significant dimensions)
            # For simplicity, we just truncate, but could implement proper PCA
            return embedding[: self.target_dim]

        else:
            # Upsizing: improved approach with scaling factor
            # Calculate scaling factor based on dimensions
            factor = self.target_dim / self.original_dim

            # For integer factor, use repeating
            if factor.is_integer():
                factor = int(factor)
                # Just repeat the embedding factor times
                repeated = np.repeat(embedding, factor)

                # If there's remaining dimensions needed, pad with zeros
                if len(repeated) < self.target_dim:
                    padding = np.zeros(self.target_dim - len(repeated))
                    return np.concatenate([repeated, padding])
                return repeated[: self.target_dim]

            else:
                # For non-integer factor, use interpolation
                # First repeat as much as possible
                repeat_times = int(factor)
                repeated = np.repeat(embedding, repeat_times)

                # Calculate how many dimensions we still need
                remaining = self.target_dim - len(repeated)

                if remaining > 0:
                    # Take the first 'remaining' values and add them at the end
                    extra = embedding[:remaining]
                    return np.concatenate([repeated, extra])

                return repeated[: self.target_dim]

    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a single text with caching
        Args:
            text: The text to embed
        Returns:
            List of floats representing the embedding (target dimensions)
        """
        try:
            # Handle empty input
            if not text or len(text.strip()) == 0:
                print_warning("Empty text provided for embedding generation")
                return [0.0] * self.target_dim  # Return zero vector

            # Check cache first if enabled
            if self.use_cache:
                cache_path = self._get_cache_path(text)
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        cached_embedding = pickle.load(f)
                        self.cache_hits += 1
                        return cached_embedding.tolist()
                self.cache_misses += 1

            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            # Resize to target dimension
            resized_embedding = self._resize_embedding(embedding)

            # Cache the embedding if enabled
            if self.use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(resized_embedding, f)

            return resized_embedding.tolist()

        except Exception as e:
            print_error(f"Error getting embedding: {str(e)}")
            print_error(f"Detailed Error Traceback: {traceback.format_exc()}")
            # Return zero vector as fallback
            return [0.0] * self.target_dim

    def get_embedding_with_chunking(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: str = "mean",
    ) -> List[float]:
        """
        Get an embedding for long text by chunking and combining
        Args:
            text: The long text to embed
            chunk_size: Maximum number of characters per chunk
            overlap: Overlap between chunks in characters
            strategy: How to combine chunks ("mean", "max", or "weighted")
        Returns:
            Combined embedding vector
        """
        if not text or len(text) <= chunk_size:
            return self.get_embedding(text)

        # Split text into overlapping chunks
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) > 20:  # Only keep substantial chunks
                chunks.append(chunk)

        if not chunks:
            return self.get_embedding(text[:chunk_size])

        # Get embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            emb = self.get_embedding(chunk)
            embeddings.append(emb)

        # Convert to numpy for operations
        embeddings_np = np.array(embeddings)

        # Combine embeddings based on strategy
        if strategy == "mean":
            combined = np.mean(embeddings_np, axis=0)
        elif strategy == "max":
            combined = np.max(embeddings_np, axis=0)
        elif strategy == "weighted":
            # Weight chunks by their length
            weights = np.array([len(chunk) for chunk in chunks])
            weights = weights / np.sum(weights)
            combined = np.sum(embeddings_np * weights[:, np.newaxis], axis=0)
        else:
            # Default to mean
            combined = np.mean(embeddings_np, axis=0)

        # Normalize the combined embedding
        if self.normalize:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

        return combined.tolist()

    def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts with optimized batching
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
        Returns:
            List of embeddings (each with target dimensions)
        """
        try:
            if not texts:
                return []

            # Filter out empty texts
            valid_texts = []
            empty_indices = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) > 0:
                    valid_texts.append(text)
                else:
                    empty_indices.append(i)

            if not valid_texts:
                print_warning("No valid texts provided for batch embedding generation")
                return [[0.0] * self.target_dim] * len(texts)

            # Check which texts are cached and which need encoding
            texts_to_encode = []
            cached_embeddings = {}

            if self.use_cache:
                for i, text in enumerate(valid_texts):
                    cache_path = self._get_cache_path(text)
                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            cached_embeddings[i] = pickle.load(f)
                            self.cache_hits += 1
                    else:
                        texts_to_encode.append((i, text))
                        self.cache_misses += 1
            else:
                texts_to_encode = [(i, text) for i, text in enumerate(valid_texts)]

            # Prepare result array with cached embeddings
            result_embeddings = [None] * len(valid_texts)
            for i, emb in cached_embeddings.items():
                result_embeddings[i] = emb.tolist()

            # Only encode texts that aren't cached
            if texts_to_encode:
                indices, encode_texts = zip(*texts_to_encode)

                # Generate embeddings with optimized batch processing
                progress_bar = tqdm(
                    desc="Generating Embeddings",
                    total=len(encode_texts),
                    disable=not show_progress,
                )

                # Process in batches to manage memory
                for i in range(0, len(encode_texts), batch_size):
                    batch_texts = encode_texts[i : i + batch_size]
                    batch_indices = indices[i : i + batch_size]

                    # Generate embeddings for batch
                    embeddings = self.model.encode(
                        batch_texts,
                        normalize_embeddings=self.normalize,
                        batch_size=batch_size,
                        show_progress_bar=False,
                    )

                    # Resize and store embeddings
                    for j, (idx, embedding) in enumerate(
                        zip(batch_indices, embeddings)
                    ):
                        resized_emb = self._resize_embedding(embedding)
                        result_embeddings[idx] = resized_emb.tolist()

                        # Cache the embedding if enabled
                        if self.use_cache:
                            cache_path = self._get_cache_path(encode_texts[i + j])
                            with open(cache_path, "wb") as f:
                                pickle.dump(resized_emb, f)

                    progress_bar.update(len(batch_texts))

                progress_bar.close()

            # Reconstruct complete result including empty texts
            final_embeddings = []
            valid_idx = 0

            for i in range(len(texts)):
                if i in empty_indices:
                    final_embeddings.append([0.0] * self.target_dim)
                else:
                    final_embeddings.append(result_embeddings[valid_idx])
                    valid_idx += 1

            return final_embeddings

        except Exception as e:
            print_error(f"Error getting batch embeddings: {str(e)}")
            print_error(f"Detailed Error Traceback: {traceback.format_exc()}")
            return [[0.0] * self.target_dim] * len(texts)

    def calculate_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two texts
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ("cosine", "euclidean", or "dot")
        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings
        emb1 = np.array(self.get_embedding(text1))
        emb2 = np.array(self.get_embedding(text2))

        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        elif method == "euclidean":
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(emb1 - emb2)
            # Convert distance to similarity (1 for identical, 0 for very different)
            return 1 / (1 + distance)

        elif method == "dot":
            # Simple dot product (assuming normalized embeddings)
            return np.dot(emb1, emb2)

        else:
            print_warning(f"Unknown similarity method: {method}, using cosine")
            # Default to cosine
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

    def search_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.6,
    ) -> List[Tuple[int, float]]:
        """
        Search for most similar documents to a query
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include in results
        Returns:
            List of (document_index, similarity_score) tuples
        """
        # Generate embedding for query
        query_embedding = np.array(self.get_embedding(query))

        # Generate embeddings for all documents
        document_embeddings = self.get_embeddings_batch(documents, show_progress=True)

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(document_embeddings):
            # Calculate cosine similarity
            doc_emb_np = np.array(doc_emb)
            dot_product = np.dot(query_embedding, doc_emb_np)
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(doc_emb_np)

            if norm1 == 0 or norm2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm1 * norm2)

            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and limit to top_k
        results = [
            (idx, score) for idx, score in similarities if score >= similarity_threshold
        ][:top_k]

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        Returns:
            Dictionary with cache statistics
        """
        if not self.use_cache:
            return {"cache_enabled": False}

        cache_files = list(Path(self.cache_dir).glob("*.pkl"))
        cache_size_bytes = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_enabled": True,
            "cache_dir": self.cache_dir,
            "cache_entries": len(cache_files),
            "cache_size_mb": round(cache_size_bytes / (1024 * 1024), 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": round(
                self.cache_hits / (self.cache_hits + self.cache_misses or 1), 2
            ),
        }

    def get_available_models(self) -> Dict[str, str]:
        """
        Get available embedding models
        Returns:
            Dictionary of available models with descriptions
        """
        return {
            "base": "all-MiniLM-L6-v2 (384d, fast)",
            "medium": "all-mpnet-base-v2 (768d, better quality)",
            "large": "all-distilroberta-v1 (768d, excellent quality)",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2 (384d, 50+ languages)",
        }
