import os
import tempfile
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import PyPDF2
import aiohttp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.colorLogger import print_info, print_error
from utils.ocr_utils import extract_text_with_ocr_fallback
from utils.vector_db import process_pdf_chunks_to_embeddings


async def download_pdf_from_cloudinary(url: str) -> bytes:
    """
    Download a PDF file from Cloudinary URL

    Args:
        url: Cloudinary URL for the PDF

    Returns:
        bytes: The PDF file content
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF: HTTP {response.status}")
                return await response.read()
    except Exception as e:
        print_error(f"Error downloading PDF (in download_pdf_from_cloudinary): {e}")
        raise


async def extract_text_from_pdf(pdf_content: bytes) -> Tuple[str, int, Dict[int, str]]:
    """
    Extract text from a PDF file

    Args:
        pdf_content: The PDF file content as bytes

    Returns:
        Tuple containing:
        - Full extracted text
        - Total number of pages
        - Dictionary mapping page numbers to page content
    """
    try:
        # Create a temporary file to save the PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name

        # Extract text using PyPDF2
        full_text = ""
        page_texts = {}

        with open(temp_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += page_text + "\n\n"
                page_texts[i + 1] = page_text  # 1-indexed page numbers

        # Check if we need OCR for any pages
        has_empty_pages = any(len(text.strip()) < 50 for text in page_texts.values())

        if has_empty_pages:
            print_info("Detected pages with little or no text, attempting OCR")
            page_texts = await extract_text_with_ocr_fallback(pdf_content, page_texts)

            # Rebuild full text from updated page texts
            full_text = ""
            for page_num in sorted(page_texts.keys()):
                full_text += page_texts[page_num] + "\n\n"

        # Clean up the temporary file
        os.unlink(temp_path)

        return full_text, total_pages, page_texts

    except Exception as e:
        print_error(f"Error extracting text from PDF (in extract_text_from_pdf): {e}")
        # Clean up the temporary file if it exists
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise


async def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(text)
        return chunks

    except Exception as e:
        print_error(f"Error chunking text (in chunk_text): {e}")
        raise


async def process_pdf(
    pdf_url: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Process a PDF file: download, extract text, and chunk it

    Args:
        pdf_url: Cloudinary URL for the PDF
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary containing:
        - full_text: The complete extracted text
        - total_pages: Number of pages in the PDF
        - chunks: List of text chunks
        - page_texts: Dictionary mapping page numbers to page content
    """
    try:
        # Download the PDF
        pdf_content = await download_pdf_from_cloudinary(pdf_url)

        # Extract text
        full_text, total_pages, page_texts = await extract_text_from_pdf(pdf_content)

        # Chunk the text
        chunks = await chunk_text(full_text, chunk_size, chunk_overlap)

        return {
            "full_text": full_text,
            "total_pages": total_pages,
            "chunks": chunks,
            "page_texts": page_texts,
        }

    except Exception as e:
        print_error(f"Error processing PDF (in process_pdf): {e}")
        raise


async def process_pdf_with_progress(
    pdf_id: str, pdf_url: str, db_pool, chunk_size: int = 1000, chunk_overlap: int = 200
) -> None:
    """
    Process a PDF file with progress tracking in the database

    Args:
        pdf_id: ID of the PDF in the database
        pdf_url: Cloudinary URL for the PDF
        db_pool: Database connection pool
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    """
    from database import (
        update_pdf_processing_status,
        update_pdf_extracted_content,
        create_pdf_chunk,
    )

    try:
        # Update status to processing
        await update_pdf_processing_status(db_pool, pdf_id, "processing", 0)

        # Download the PDF
        pdf_content = await download_pdf_from_cloudinary(pdf_url)
        await update_pdf_processing_status(db_pool, pdf_id, "processing", 20)

        # Extract text
        full_text, total_pages, page_texts = await extract_text_from_pdf(pdf_content)
        await update_pdf_processing_status(
            db_pool, pdf_id, "processing", 50, total_pages
        )

        # Store full text (optional, can be skipped if too large)
        if len(full_text) < 10 * 1024 * 1024:  # Only store if less than 10MB
            await update_pdf_extracted_content(db_pool, pdf_id, full_text)
        else:
            print_info(f"Skipping full text storage for PDF {pdf_id} (too large)")

        # Chunk the text
        chunks = await chunk_text(full_text, chunk_size, chunk_overlap)
        await update_pdf_processing_status(db_pool, pdf_id, "processing", 70)

        # Store chunks in database
        for i, chunk_content in enumerate(chunks):
            # Determine which page this chunk likely belongs to
            # This is a simple approach - more sophisticated mapping would be needed for accuracy
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
                db_pool, pdf_id, i, chunk_content, page_number, metadata  # chunk_index
            )

        # Generate embeddings for the chunks
        try:
            # Start embedding generation
            print_info(f"Starting embedding generation for PDF {pdf_id}")
            await process_pdf_chunks_to_embeddings(db_pool, pdf_id)
        except Exception as embed_error:
            print_error(
                f"Error generating embeddings for PDF {pdf_id} (in process_pdf_with_progress): {embed_error}"
            )
            # Continue processing even if embedding fails
            # The PDF is still usable without embeddings
            await update_pdf_processing_status(
                db_pool,
                pdf_id,
                "completed_without_embeddings",
                100,
                None,
                f"Completed processing but embeddings failed: {str(embed_error)}",
            )
            return

        # Update status to completed
        await update_pdf_processing_status(db_pool, pdf_id, "completed", 100)
        print_info(f"PDF processing completed for PDF {pdf_id}")

    except Exception as e:
        print_error(
            f"Error processing PDF {pdf_id} (in process_pdf_with_progress): {e}"
        )
        # Update status to failed
        await update_pdf_processing_status(
            db_pool, pdf_id, "failed", None, None, str(e)
        )
        raise
