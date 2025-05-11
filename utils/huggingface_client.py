import os
import json
import aiohttp
import numpy as np
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
from utils.colorLogger import print_info, print_error

# Load environment variables
load_dotenv()

# Get Hugging Face API token from environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    print_error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Corrected API URL format
HF_API_BASE_URL = "https://api-inference.huggingface.co/models"


class AsyncHuggingFaceClient:
    """
    Asynchronous client for the Hugging Face API
    """

    def __init__(self, api_token: str = None, model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the client

        Args:
            api_token: Hugging Face API token (defaults to environment variable)
            model: Default model to use
        """
        self.api_token = api_token or HUGGINGFACEHUB_API_TOKEN
        if not self.api_token:
            raise ValueError("Hugging Face API token is required")

        self.model = model
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Get an embedding for a single text

        Args:
            text: The text to embed
            model: Model to use (defaults to instance default)

        Returns:
            List of floats representing the embedding
        """
        if not text:
            return []

        model = model or self.model
        url = f"{HF_API_BASE_URL}/{model}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print_error(f"Error from Hugging Face API: {error_text}")
                    raise Exception(f"Hugging Face API error: {response.status}")

                embedding = await response.json()

                # For sentence-transformers models, the response is already a list of floats
                # For some models, we might need to handle pooling of token embeddings
                if isinstance(embedding, list) and isinstance(embedding[0], list):
                    # Average pooling for token-level embeddings
                    embedding = np.mean(embedding, axis=0).tolist()

                return embedding

    async def get_embeddings_batch(
        self, texts: List[str], model: str = None
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to instance default)

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        model = model or self.model
        url = f"{HF_API_BASE_URL}/{model}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print_error(f"Error from Hugging Face API: {error_text}")
                    raise Exception(f"Hugging Face API error: {response.status}")

                embeddings = await response.json()

                # For sentence-transformers models, the response is already a list of lists of floats
                # For some models, we might need to handle pooling of token embeddings
                if isinstance(embeddings, list) and all(
                    isinstance(emb, list) for emb in embeddings
                ):
                    if all(
                        isinstance(item, list) for emb in embeddings for item in emb
                    ):
                        # Token-level embeddings need pooling
                        embeddings = [
                            np.mean(emb, axis=0).tolist() for emb in embeddings
                        ]

                return embeddings
