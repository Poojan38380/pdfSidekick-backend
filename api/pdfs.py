from fastapi import APIRouter, HTTPException, status, Request, UploadFile, File, Form
from typing import Dict, Any, List
import uuid
from database import get_pdfs_by_user_id, get_pdf_by_id, get_user_by_id, create_pdf
from schemas import PDFResponse
from utils.cloudinary_utils import upload_pdf_to_cloudinary
from utils.colorLogger import print_error, print_info

# Configure logging

router = APIRouter()


@router.post("/upload", response_model=PDFResponse)
async def create_new_pdf(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(None),
    user_id: str = Form(...),
) -> PDFResponse:
    """
    Uploads the PDF to Cloudinary and saves the metadata to the database.
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
            print_info(upload_result)
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
