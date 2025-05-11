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

# Default embedding model - using a model specifically for embeddings/feature extraction
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Hugging Face Inference API URL
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
        print_info(f"Initialized Hugging Face client with model: {self.model}")

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Get an embedding for a single text
        Args:
            text: The text to embed
            model: Model to use (defaults to instance default)
        Returns:
            List of floats representing the embedding
        """
        embeddings = await self.get_embeddings_batch([text], model)
        return embeddings[0] if embeddings else []

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

        model_name = model or self.model
        url = f"{HF_API_BASE_URL}/{model_name}"

        # Maximum number of texts to send in a single batch
        # For longer texts or large batches, processing can time out
        MAX_BATCH_SIZE = 8

        try:
            async with aiohttp.ClientSession() as session:
                all_embeddings = []

                # Process texts in batches to avoid timeouts
                for i in range(0, len(texts), MAX_BATCH_SIZE):
                    batch = texts[i : i + MAX_BATCH_SIZE]

                    # Different models require different payload formats
                    if "sentence-transformers" in model_name:
                        # Process texts one by one for sentence-transformers
                        batch_embeddings = []
                        for text in batch:
                            # For feature extraction (not similarity)
                            payload = {"inputs": text, "task": "feature-extraction"}
                            print_info(
                                f"Sending request to {url} with feature-extraction task"
                            )

                            async with session.post(
                                url, headers=self.headers, json=payload
                            ) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print_error(
                                        f"Error from Hugging Face API: {error_text}"
                                    )
                                    response.raise_for_status()

                                result = await response.json()
                                batch_embeddings.append(result)

                        all_embeddings.extend(batch_embeddings)
                    else:
                        # For embedding models like BGE
                        payload = {"inputs": batch, "options": {"wait_for_model": True}}
                        print_info(
                            f"Sending batch of {len(batch)} texts to {model_name}"
                        )

                        async with session.post(
                            url, headers=self.headers, json=payload
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                print_error(
                                    f"Error from Hugging Face API: {error_text}"
                                )
                                response.raise_for_status()

                            result = await response.json()

                            # Handle different response formats
                            if isinstance(result, list):
                                if len(result) > 0:
                                    if isinstance(result[0], list):
                                        # Direct embeddings list
                                        all_embeddings.extend(result)
                                    elif isinstance(result[0], (dict, float)):
                                        # Could be embedding in dict or a single vector
                                        if (
                                            isinstance(result[0], dict)
                                            and "embedding" in result[0]
                                        ):
                                            all_embeddings.extend(
                                                [item["embedding"] for item in result]
                                            )
                                        else:
                                            # Might be a single embedding returned as list of floats
                                            all_embeddings.append(result)
                            elif isinstance(result, dict):
                                if "embeddings" in result:
                                    all_embeddings.extend(result["embeddings"])
                                elif "embedding" in result:
                                    all_embeddings.append(result["embedding"])
                            else:
                                print_error(f"Unexpected response format: {result}")

                return all_embeddings

        except aiohttp.ClientError as e:
            print_error(f"Network error when calling Hugging Face API: {str(e)}")
            return []
        except Exception as e:
            print_error(f"Error getting embeddings: {str(e)}")
            return []

    async def normalize_embeddings(
        self, embeddings: List[List[float]]
    ) -> List[List[float]]:
        """
        Normalize the embeddings to unit length
        Args:
            embeddings: List of embeddings to normalize
        Returns:
            List of normalized embeddings
        """
        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append((np.array(emb) / norm).tolist())
            else:
                normalized.append(emb)  # Keep as-is if zero vector
        return normalized
