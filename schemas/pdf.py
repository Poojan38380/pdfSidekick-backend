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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: str
    
    class Config:
        from_attributes = True 