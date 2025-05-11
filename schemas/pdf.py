from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class PDFBase(BaseModel):
    title: str
    description: Optional[str] = None
    document_link: str


class PDFCreate(PDFBase):
    user_id: str


class PDFResponse(PDFBase):
    id: str
    created_at: datetime
    updated_at: datetime
    user_id: str
    processing_status: Optional[str] = "pending"
    processing_progress: Optional[float] = 0
    total_pages: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {uuid.UUID: str}


class PDFChunkBase(BaseModel):
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class PDFChunkCreate(PDFChunkBase):
    pdf_id: str


class PDFChunkResponse(PDFChunkBase):
    id: str
    created_at: datetime
    pdf_id: str

    class Config:
        from_attributes = True
        json_encoders = {uuid.UUID: str}
