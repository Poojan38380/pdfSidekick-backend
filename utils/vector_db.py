import os
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from dotenv import load_dotenv
from utils.colorLogger import print_info, print_error
from utils.local_embedding_client import (
    LocalEmbeddingClient,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
)

# Load environment variables
load_dotenv()

# Initialize the Local Embedding client
embedding_client = LocalEmbeddingClient()


async def generate_embedding(
    text: str, model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
) -> List[float]:
    """
    Generate an embedding for the given text using local embedding model

    Args:
        text: The text to generate an embedding for
        model: The embedding model to use

    Returns:
        List of floats representing the embedding vector (1536 dimensions)
    """
    try:
        if not text or len(text.strip()) == 0:
            print_error("Empty text provided for embedding generation")
            return []

        return embedding_client.get_embedding(text)
    except Exception as e:
        print_error(f"Error generating embedding (in generate_embedding): {e}")
        raise


async def generate_embeddings_batch(
    texts: List[str], model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a batch

    Args:
        texts: List of texts to generate embeddings for
        model: The embedding model to use

    Returns:
        List of embedding vectors (each with 1536 dimensions)
    """
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if text and len(text.strip()) > 0]
        if not valid_texts:
            print_error("No valid texts provided for batch embedding generation")
            return []

        # Process in smaller batches to avoid memory issues
        batch_size = 32  # Adjust based on available memory
        all_embeddings = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]

            # Generate embeddings for the batch
            try:
                batch_embeddings = embedding_client.get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print_error(
                    f"Error in batch {i//batch_size + 1} (in generate_embeddings_batch): {e}"
                )
                # Continue with next batch instead of failing completely
                continue

            # Add a small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        return all_embeddings
    except Exception as e:
        print_error(
            f"Error generating batch embeddings (in generate_embeddings_batch): {e}"
        )
        raise


async def process_pdf_chunks_to_embeddings(
    db_pool, pdf_id: str, model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
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
        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "embedding",
            80,
            None,
            None,
            "generating_embeddings",
            len(chunks),
            0,
        )

        # Process chunks in batches
        batch_size = 32  # LangChain can handle larger batches
        total_chunks = len(chunks)
        embeddings_created = 0

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            texts = [chunk["content"] for chunk in batch_chunks]

            try:
                # Generate embeddings for the batch
                embeddings = await generate_embeddings_batch(texts, model)

                # Verify we got the expected number of embeddings
                if len(embeddings) != len(batch_chunks):
                    print_error(
                        f"Mismatch in batch {i//batch_size + 1}: got {len(embeddings)} embeddings for {len(batch_chunks)} chunks"
                    )
                    continue

                # Store embeddings in the database
                for j, embedding in enumerate(embeddings):
                    chunk = batch_chunks[j]
                    try:
                        await create_pdf_embedding(
                            db_pool, pdf_id, chunk["id"], embedding, model
                        )
                        embeddings_created += 1
                    except Exception as db_error:
                        print_error(
                            f"Error storing embedding for chunk {chunk['id']} (in process_pdf_chunks_to_embeddings): {db_error}"
                        )
                        continue
            except Exception as batch_error:
                print_error(
                    f"Error processing batch {i//batch_size + 1} (in process_pdf_chunks_to_embeddings): {batch_error}"
                )
                continue

            # Update progress
            progress = 80 + (20 * (i + len(batch_chunks)) / total_chunks)
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "embedding",
                min(progress, 99),  # Cap at 99% until fully complete
                None,
                None,
                "generating_embeddings",
                total_chunks,
                embeddings_created,
            )

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

        # Update status based on how many embeddings were created
        if embeddings_created == 0:
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "failed",
                None,
                None,
                "Failed to generate any embeddings",
                "embedding_failed",
                total_chunks,
                0,
            )
            return {
                "status": "failed",
                "pdf_id": pdf_id,
                "chunks_processed": total_chunks,
                "embeddings_created": 0,
            }
        elif embeddings_created < total_chunks:
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "completed_partial",
                100,
                None,
                f"Generated {embeddings_created}/{total_chunks} embeddings",
                "embedding_partial",
                total_chunks,
                embeddings_created,
            )
        else:
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "completed",
                100,
                None,
                None,
                "embedding_completed",
                total_chunks,
                embeddings_created,
            )

        print_info(
            f"Embedding generation completed for PDF {pdf_id}: {embeddings_created}/{total_chunks} embeddings created"
        )

        return {
            "status": "completed",
            "pdf_id": pdf_id,
            "chunks_processed": total_chunks,
            "embeddings_created": embeddings_created,
        }

    except Exception as e:
        print_error(
            f"Error processing PDF chunks to embeddings (in process_pdf_chunks_to_embeddings): {e}"
        )
        # Update status to failed
        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "failed",
            None,
            None,
            f"Embedding generation failed: {str(e)}",
            "embedding_error",
            None,
            None,
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

        if not query_embedding:
            print_error("Failed to generate embedding for query")
            return []

        # Search for similar chunks
        results = await search_similar_chunks(
            db_pool, query_embedding, limit, threshold
        )

        print_info(f"Semantic search for '{query}' found {len(results)} results")
        return results

    except Exception as e:
        print_error(f"Error performing semantic search (in semantic_search): {e}")
        raise
