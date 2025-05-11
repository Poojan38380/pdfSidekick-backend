from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Request,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Query,
)
from typing import Dict, Any, List
import uuid
from database import (
    get_pdfs_by_user_id,
    get_pdf_by_id,
    get_user_by_id,
    create_pdf,
    get_pdf_chunks,
    update_pdf_processing_status,
)
from schemas import PDFResponse, PDFChunkResponse, SearchResponse
from utils.cloudinary_utils import upload_pdf_to_cloudinary
from utils.colorLogger import print_error, print_info
from utils.pdf_processor import process_pdf_with_progress
from utils.vector_db import semantic_search, process_pdf_chunks_to_embeddings

# Configure logging

router = APIRouter()


@router.post("/upload", response_model=PDFResponse)
async def create_new_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(None),
    user_id: str = Form(...),
) -> PDFResponse:
    """
    Uploads the PDF to Cloudinary, saves the metadata to the database,
    and triggers background processing of the PDF content.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed"
        )

    pool = request.app.state.db_pool
    user = await get_user_by_id(pool, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    try:
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded"
            )

        try:
            upload_result = await upload_pdf_to_cloudinary(file_content)
            document_link = upload_result["secure_url"]

        except Exception as cloud_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error uploading to cloud storage: {str(cloud_error)}",
            )

        try:
            pdf_data = await create_pdf(
                pool=pool,
                title=title,
                description=description,
                document_link=document_link,
                user_id=user_id,
            )

            # Start background processing of the PDF
            background_tasks.add_task(
                process_pdf_with_progress, pdf_data["id"], document_link, pool
            )

            print_info(f"PDF {pdf_data['id']} uploaded and queued for processing")
            return pdf_data

        except Exception as db_error:
            print_error(db_error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving to database: {str(db_error)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading PDF: {str(e)}",
        )


@router.get("/user/{user_id}", response_model=List[PDFResponse])
async def get_user_pdfs(user_id: str, request: Request) -> List[Dict[str, Any]]:
    """
    Get all PDFs for a specific user
    """
    pool = request.app.state.db_pool

    try:

        user = await get_user_by_id(pool, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )
        pdfs = await get_pdfs_by_user_id(pool, user_id)
        return pdfs
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDFs: {str(e)}",
        )


@router.get("/{pdf_id}", response_model=PDFResponse)
async def get_pdf_details(pdf_id: str, request: Request) -> Dict[str, Any]:
    """
    Get details of a specific PDF
    """
    pool = request.app.state.db_pool

    try:
        pdf = await get_pdf_by_id(pool, pdf_id)

        if not pdf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found"
            )

        return pdf
    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDF: {str(e)}",
        )


@router.get("/{pdf_id}/chunks", response_model=List[PDFChunkResponse])
async def get_pdf_text_chunks(pdf_id: str, request: Request) -> List[Dict[str, Any]]:
    """
    Get all text chunks for a specific PDF
    """
    pool = request.app.state.db_pool

    try:
        # First check if the PDF exists
        pdf = await get_pdf_by_id(pool, pdf_id)
        if not pdf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found"
            )

        # Check processing status
        if (
            pdf.get("processing_status") == "pending"
            or pdf.get("processing_status") == "processing"
        ):
            raise HTTPException(
                status_code=status.HTTP_102_PROCESSING,
                detail=f"PDF is still being processed. Current progress: {pdf.get('processing_progress', 0)}%",
            )
        elif pdf.get("processing_status") == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF processing failed: {pdf.get('error_message', 'Unknown error')}",
            )

        # Get chunks
        chunks = await get_pdf_chunks(pool, pdf_id)
        return chunks

    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDF chunks: {str(e)}",
        )


@router.get("/{pdf_id}/processing-status")
async def get_pdf_processing_status(pdf_id: str, request: Request) -> Dict[str, Any]:
    """
    Get the processing status of a specific PDF
    """
    pool = request.app.state.db_pool

    try:
        # Check if the PDF exists
        pdf = await get_pdf_by_id(pool, pdf_id)
        if not pdf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found"
            )

        # Return processing status information
        return {
            "id": pdf["id"],
            "processing_status": pdf.get("processing_status", "pending"),
            "processing_progress": pdf.get("processing_progress", 0),
            "total_pages": pdf.get("total_pages"),
            "error_message": pdf.get("error_message"),
        }

    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving processing status: {str(e)}",
        )


@router.post("/{pdf_id}/reprocess")
async def reprocess_pdf(
    pdf_id: str, request: Request, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Reprocess a PDF that may have failed or needs to be processed again
    """
    pool = request.app.state.db_pool

    try:
        # Check if the PDF exists
        pdf = await get_pdf_by_id(pool, pdf_id)
        if not pdf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found"
            )

        # Reset processing status
        updated_pdf = await update_pdf_processing_status(
            pool, pdf_id, "pending", 0, None, None
        )

        # Start background processing
        background_tasks.add_task(
            process_pdf_with_progress, pdf_id, pdf["document_link"], pool
        )

        print_info(f"PDF {pdf_id} queued for reprocessing")

        return {
            "id": updated_pdf["id"],
            "processing_status": updated_pdf["processing_status"],
            "processing_progress": updated_pdf["processing_progress"],
            "message": "PDF queued for reprocessing",
        }

    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing PDF: {str(e)}",
        )


@router.post("/{pdf_id}/generate-embeddings")
async def generate_pdf_embeddings(
    pdf_id: str, request: Request, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate embeddings for a PDF that has been processed
    """
    pool = request.app.state.db_pool

    try:
        # Check if the PDF exists
        pdf = await get_pdf_by_id(pool, pdf_id)
        if not pdf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found"
            )

        # Check if the PDF has been processed
        if pdf.get("processing_status") not in [
            "completed",
            "completed_without_embeddings",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF must be fully processed before generating embeddings. Current status: {pdf.get('processing_status')}",
            )

        # Update processing status to embedding
        updated_pdf = await update_pdf_processing_status(
            pool, pdf_id, "embedding", 80, None, None
        )

        # Start background embedding generation
        background_tasks.add_task(process_pdf_chunks_to_embeddings, pool, pdf_id)

        print_info(f"PDF {pdf_id} queued for embedding generation")

        return {
            "id": updated_pdf["id"],
            "processing_status": updated_pdf["processing_status"],
            "processing_progress": updated_pdf["processing_progress"],
            "message": "PDF queued for embedding generation",
        }

    except HTTPException:
        raise
    except Exception as e:
        print_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}",
        )


@router.get("/search")
async def search_pdfs(
    request: Request,
    query: str,
    limit: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.7, ge=0, le=1.0),
) -> SearchResponse:
    """
    Search PDFs using semantic search
    """
    pool = request.app.state.db_pool

    try:
        # Perform semantic search
        results = await semantic_search(pool, query, limit, threshold)

        # Format the results
        search_results = []
        for result in results:
            search_results.append(
                {
                    "pdf_id": str(result["pdf_id"]),
                    "pdf_title": result["title"],
                    "chunk_id": str(result["chunk_id"]),
                    "content": result["content"],
                    "page_number": result["page_number"],
                    "similarity": result["similarity"],
                    "metadata": result["metadata"] if result["metadata"] else {},
                }
            )

        return {"query": query, "results": search_results, "count": len(search_results)}

    except Exception as e:
        print_error(f"Error performing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}",
        )
