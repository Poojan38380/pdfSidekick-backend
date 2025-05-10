from pydantic import BaseModel, Field
from typing import Optional
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
    
    class Config:
        from_attributes = True
        json_encoders = {
            uuid.UUID: str
        } 