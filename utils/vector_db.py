import os
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from dotenv import load_dotenv
from utils.colorLogger import print_info, print_error
from utils.huggingface_client import AsyncHuggingFaceClient, DEFAULT_EMBEDDING_MODEL

# Load environment variables
load_dotenv()

# Get Hugging Face API token from environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    print_error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

# Initialize the Hugging Face client
hf_client = AsyncHuggingFaceClient(api_token=HUGGINGFACEHUB_API_TOKEN)


async def generate_embedding(
    text: str, model: str = DEFAULT_EMBEDDING_MODEL
) -> List[float]:
    """
    Generate an embedding for the given text using Hugging Face's API

    Args:
        text: The text to generate an embedding for
        model: The Hugging Face embedding model to use

    Returns:
        List of floats representing the embedding vector
    """
    try:
        return await hf_client.get_embedding(text, model)
    except Exception as e:
        print_error(f"Error generating embedding: {e}")
        raise


async def generate_embeddings_batch(
    texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a batch

    Args:
        texts: List of texts to generate embeddings for
        model: The Hugging Face embedding model to use

    Returns:
        List of embedding vectors
    """
    try:
        # Process in smaller batches to avoid overloading the API
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Generate embeddings for the batch
            batch_embeddings = await hf_client.get_embeddings_batch(batch, model)
            all_embeddings.extend(batch_embeddings)

            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return all_embeddings
    except Exception as e:
        print_error(f"Error generating batch embeddings: {e}")
        raise


async def process_pdf_chunks_to_embeddings(
    db_pool, pdf_id: str, model: str = DEFAULT_EMBEDDING_MODEL
) -> Dict[str, Any]:
    """
    Process all chunks of a PDF and generate embeddings

    Args:
        db_pool: Database connection pool
        pdf_id: ID of the PDF to process
        model: Embedding model to use

    Returns:
        Dictionary with processing results
    """
    from database import (
        get_pdf_chunks,
        create_pdf_embedding,
        update_pdf_processing_status,
    )

    try:
        # Get all chunks for the PDF
        chunks = await get_pdf_chunks(db_pool, pdf_id)

        if not chunks:
            print_info(f"No chunks found for PDF {pdf_id}")
            return {"status": "no_chunks", "pdf_id": pdf_id, "chunks_processed": 0}

        # Update processing status
        await update_pdf_processing_status(db_pool, pdf_id, "embedding", 80, None, None)

        # Process chunks in batches to avoid rate limits
        batch_size = 8  # Smaller batch size for Hugging Face API
        total_chunks = len(chunks)
        embeddings_created = 0

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            texts = [chunk["content"] for chunk in batch_chunks]

            # Generate embeddings for the batch
            embeddings = await generate_embeddings_batch(texts, model)

            # Store embeddings in the database
            for j, embedding in enumerate(embeddings):
                chunk = batch_chunks[j]
                await create_pdf_embedding(
                    db_pool, pdf_id, chunk["id"], embedding, model
                )
                embeddings_created += 1

            # Update progress
            progress = 80 + (20 * (i + len(batch_chunks)) / total_chunks)
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "embedding",
                min(progress, 99),  # Cap at 99% until fully complete
                None,
                None,
            )

            # Small delay to avoid overwhelming the API
            await asyncio.sleep(1.0)  # Longer delay for Hugging Face API

        # Update status to completed
        await update_pdf_processing_status(db_pool, pdf_id, "completed", 100)

        return {
            "status": "completed",
            "pdf_id": pdf_id,
            "chunks_processed": total_chunks,
            "embeddings_created": embeddings_created,
        }

    except Exception as e:
        print_error(f"Error processing PDF chunks to embeddings: {e}")
        # Update status to failed
        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "failed",
            None,
            None,
            f"Embedding generation failed: {str(e)}",
        )
        raise


async def semantic_search(
    db_pool, query: str, limit: int = 5, threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using the query text

    Args:
        db_pool: Database connection pool
        query: The search query text
        limit: Maximum number of results to return
        threshold: Minimum similarity threshold (0-1)

    Returns:
        List of similar chunks with similarity scores
    """
    from database import search_similar_chunks

    try:
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)

        # Search for similar chunks
        results = await search_similar_chunks(
            db_pool, query_embedding, limit, threshold
        )

        return results

    except Exception as e:
        print_error(f"Error performing semantic search: {e}")
        raise
