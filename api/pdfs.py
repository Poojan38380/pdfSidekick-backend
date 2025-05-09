from fastapi import APIRouter, HTTPException, status, Request
from typing import Dict, Any, List

from database import  create_pdf, get_pdfs_by_user_id, get_pdf_by_id
from schemas import PDFCreate, PDFResponse

router = APIRouter()

@router.post("/", response_model=PDFResponse, status_code=status.HTTP_201_CREATED)
async def create_new_pdf(pdf_data: PDFCreate, request: Request) -> Dict[str, Any]:
    """
    Create a new PDF document entry
    """
    pool = request.app.state.db_pool
    
    try:
        new_pdf = await create_pdf(
            pool=pool,
            title=pdf_data.title,
            description=pdf_data.description,
            document_link=pdf_data.document_link,
            user_id=pdf_data.user_id
        )
        return new_pdf
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating PDF: {str(e)}"
        )

@router.get("/user/{user_id}", response_model=List[PDFResponse])
async def get_user_pdfs(user_id: str, request: Request) -> List[Dict[str, Any]]:
    """
    Get all PDFs for a specific user
    """
    pool = request.app.state.db_pool
    
    try:
        pdfs = await get_pdfs_by_user_id(pool, user_id)
        return pdfs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDFs: {str(e)}"
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
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF not found"
            )
        
        return pdf
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDF: {str(e)}"
        ) 