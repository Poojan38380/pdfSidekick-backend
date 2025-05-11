import os
import json
import aiohttp
import asyncio
import numpy as np
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
from utils.colorLogger import print_info, print_error
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

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
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        print_info(f"Initialized Hugging Face client with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _make_request(self, url: str, payload: dict) -> Any:
        """
        Make a request to the Hugging Face API with retry logic

        Args:
            url: API endpoint URL
            payload: Request payload

        Returns:
            API response
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=self.headers, json=payload, timeout=self.timeout
                ) as response:
                    if response.status == 504:
                        print_error(
                            f"Gateway timeout from Hugging Face API, retrying..."
                        )
                        raise tenacity.TryAgain()

                    if response.status != 200:
                        error_text = await response.text()
                        print_error(f"Error from Hugging Face API: {error_text}")
                        raise Exception(f"Hugging Face API error: {response.status}")

                    return await response.json()
        except asyncio.TimeoutError:
            print_error("Request timed out, retrying...")
            print_error(
                f"Error making request to Hugging Face API (in _make_request): {e}"
            )
            raise tenacity.TryAgain()
        except Exception as e:
            print_error(
                f"Error making request to Hugging Face API (in _make_request): {e}"
            )
            if isinstance(e, tenacity.TryAgain):
                raise
            print_error(f"Request failed: {str(e)}")
            raise

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

        model_name = model or self.model
        url = f"{HF_API_BASE_URL}/{model_name}"

        # Prepare the payload with the correct format
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True, "use_cache": True},
        }

        try:
            result = await self._make_request(url, payload)

            # Handle different response formats
            if isinstance(result, list):
                if isinstance(result[0], list):
                    # Token-level embeddings need pooling
                    result = np.mean(result, axis=0).tolist()
                elif len(result) == 1 and isinstance(result[0], (list, np.ndarray)):
                    # Single embedding in a list
                    result = result[0]

            return result
        except Exception as e:
            print_error(f"Error getting embedding (in get_embedding): {str(e)}")
            raise

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
        MAX_BATCH_SIZE = 4  # Reduced batch size to prevent timeouts

        try:
            all_embeddings = []

            # Process texts in batches to avoid timeouts
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch = texts[i : i + MAX_BATCH_SIZE]

                # Prepare payload
                payload = {
                    "inputs": batch,
                    "options": {"wait_for_model": True, "use_cache": True},
                }

                print_info(f"Sending batch of {len(batch)} texts to {model_name}")

                result = await self._make_request(url, payload)

                # Handle different response formats
                if isinstance(result, list):
                    if len(result) > 0:
                        if isinstance(result[0], list):
                            # Direct embeddings list
                            all_embeddings.extend(result)
                        elif isinstance(result[0], (dict, float)):
                            # Could be embedding in dict or a single vector
                            if isinstance(result[0], dict) and "embedding" in result[0]:
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

                # Add a small delay to avoid rate limiting
                await asyncio.sleep(1.0)

            return all_embeddings

        except Exception as e:
            print_error(
                f"Error getting batch embeddings (in get_embeddings_batch): {str(e)}"
            )
            raise

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
