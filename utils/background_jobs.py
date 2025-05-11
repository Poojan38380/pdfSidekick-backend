import asyncio
import traceback
from typing import Dict, Any, List, Optional
from utils.colorLogger import print_info, print_error
from utils.pdf_processor import process_pdf
from utils.vector_db import process_pdf_chunks_to_embeddings
from database import (
    update_pdf_processing_status,
    create_pdf_chunk,
    update_pdf_extracted_content,
    get_pdf_by_id,
)

# Queue for background jobs
processing_jobs = {}


async def index_pdf_document(
    db_pool, pdf_id: str, pdf_url: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> None:
    """
    Complete indexing workflow for a PDF document:
    1. Extract text from PDF
    2. Chunk the content
    3. Generate embeddings
    4. Store in vector DB

    Args:
        db_pool: Database connection pool
        pdf_id: ID of the PDF to process
        pdf_url: URL of the PDF document
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    job_id = f"pdf-indexing-{pdf_id}"

    # Register job in tracking system
    processing_jobs[job_id] = {
        "status": "processing",
        "pdf_id": pdf_id,
        "progress": 0,
        "step": "starting",
    }

    try:
        # Update status to processing
        await update_pdf_processing_status(
            db_pool, pdf_id, "processing", 0, None, None, "starting", 0, 0
        )

        # Step 1: Extract text and chunk content
        processing_jobs[job_id]["step"] = "extracting_text"
        print_info(f"[Job {job_id}] Extracting text from PDF {pdf_id}")

        pdf_data = await process_pdf(pdf_url, chunk_size, chunk_overlap)
        total_pages = pdf_data.get("total_pages", 0)
        chunks = pdf_data.get("chunks", [])
        full_text = pdf_data.get("full_text", "")
        page_texts = pdf_data.get("page_texts", {})

        # Update progress
        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "processing",
            30,
            total_pages,
            None,
            "extracting_text",
            0,
            0,
        )
        processing_jobs[job_id]["progress"] = 30

        # Store full text if not too large
        if full_text and len(full_text) < 10 * 1024 * 1024:  # 10MB limit
            await update_pdf_extracted_content(db_pool, pdf_id, full_text)

        # Step 2: Store chunks in database
        processing_jobs[job_id]["step"] = "storing_chunks"
        print_info(f"[Job {job_id}] Storing {len(chunks)} chunks for PDF {pdf_id}")

        chunks_stored = 0
        for i, chunk_content in enumerate(chunks):
            # Determine which page this chunk likely belongs to
            page_number = None
            for page_num, page_text in page_texts.items():
                if chunk_content in page_text:
                    page_number = page_num
                    break

            # Create metadata
            metadata = {
                "chunk_size": len(chunk_content),
                "page_number": page_number,
                "chunk_index": i,
            }

            # Store chunk
            await create_pdf_chunk(
                db_pool, pdf_id, i, chunk_content, page_number, metadata
            )
            chunks_stored += 1

            # Update progress periodically
            if i % 10 == 0 or i == len(chunks) - 1:
                chunk_progress = 30 + (30 * (i + 1) / len(chunks))
                await update_pdf_processing_status(
                    db_pool,
                    pdf_id,
                    "processing",
                    chunk_progress,
                    total_pages,
                    None,
                    "storing_chunks",
                    chunks_stored,
                    0,
                )

        # Update progress
        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "processing",
            60,
            total_pages,
            None,
            "chunks_completed",
            chunks_stored,
            0,
        )
        processing_jobs[job_id]["progress"] = 60

        # Step 3 & 4: Generate embeddings and store in vector DB
        processing_jobs[job_id]["step"] = "generating_embeddings"
        print_info(f"[Job {job_id}] Generating embeddings for PDF {pdf_id}")

        embedding_result = await process_pdf_chunks_to_embeddings(db_pool, pdf_id)
        embeddings_created = embedding_result.get("embeddings_created", 0)

        # Final status update
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["step"] = "completed"

        await update_pdf_processing_status(
            db_pool,
            pdf_id,
            "completed",
            100,
            total_pages,
            None,
            "completed",
            chunks_stored,
            embeddings_created,
        )

        print_info(
            f"[Job {job_id}] Indexing completed for PDF {pdf_id}: {chunks_stored} chunks, {embeddings_created} embeddings"
        )

    except Exception as e:
        error_msg = f"Error indexing PDF {pdf_id} (in index_pdf_document): {str(e)}\n{traceback.format_exc()}"
        print_error(error_msg)

        # Update status to failed
        await update_pdf_processing_status(
            db_pool, pdf_id, "failed", None, None, error_msg, "failed", None, None
        )

        # Update job tracking
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)

    finally:
        # Keep job info for a while, then clean up
        asyncio.create_task(cleanup_job(job_id))


async def cleanup_job(job_id: str, delay: int = 3600) -> None:
    """
    Clean up job tracking after a delay

    Args:
        job_id: ID of the job to clean up
        delay: Delay in seconds before cleanup (default: 1 hour)
    """
    await asyncio.sleep(delay)
    if job_id in processing_jobs:
        del processing_jobs[job_id]
        print_info(f"Cleaned up job tracking for {job_id}")


async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a background job

    Args:
        job_id: ID of the job

    Returns:
        Job status information or None if job not found
    """
    return processing_jobs.get(job_id)


async def get_pdf_indexing_status(db_pool, pdf_id: str) -> Dict[str, Any]:
    """
    Get the indexing status of a PDF document

    Args:
        db_pool: Database connection pool
        pdf_id: ID of the PDF

    Returns:
        PDF indexing status information
    """
    # Get PDF details from database
    pdf = await get_pdf_by_id(db_pool, pdf_id)

    if not pdf:
        return {"pdf_id": pdf_id, "status": "not_found", "progress": 0}

    # Check if there's an active job
    job_id = f"pdf-indexing-{pdf_id}"
    job_info = processing_jobs.get(job_id)

    # Return combined status information
    return {
        "pdf_id": pdf_id,
        "status": pdf.get("processing_status", "unknown"),
        "progress": pdf.get("processing_progress", 0),
        "total_pages": pdf.get("total_pages"),
        "error_message": pdf.get("error_message"),
        "active_job": job_info is not None,
        "job_step": job_info.get("step") if job_info else None,
    }
