import os
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from utils.colorLogger import print_info, print_error

# Default embedding model - using a model specifically for embeddings/feature extraction
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class LangChainEmbeddingClient:
    """
    Client for handling embeddings using LangChain
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the LangChain embedding client
        Args:
            model_name: The name of the embedding model to use
        """
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print_info(
                f"Initialized LangChain embedding client with model: {model_name}"
            )
        except Exception as e:
            print_error(f"Error initializing LangChain client: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a single text
        Args:
            text: The text to embed
        Returns:
            List of floats representing the embedding
        """
        try:
            if not text or len(text.strip()) == 0:
                print_error("Empty text provided for embedding generation")
                return []

            embedding = self.model.embed_query(text)
            return embedding
        except Exception as e:
            print_error(f"Error getting embedding: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts
        Args:
            texts: List of texts to embed
        Returns:
            List of embeddings
        """
        try:
            if not texts:
                return []

            # Filter out empty texts
            valid_texts = [text for text in texts if text and len(text.strip()) > 0]
            if not valid_texts:
                print_error("No valid texts provided for batch embedding generation")
                return []

            embeddings = self.model.embed_documents(valid_texts)
            return embeddings
        except Exception as e:
            print_error(f"Error getting batch embeddings: {e}")
            raise
