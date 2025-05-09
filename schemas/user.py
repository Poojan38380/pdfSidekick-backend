from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
import uuid

class UserBase(BaseModel):
    id: str
    username: str
    first_name: str
    last_name: str
    email: str


class UserResponse(UserBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    profile_pic: str = "https://ui-avatars.com/api/?background=random&name=x&bold=true"
    
    class Config:
        from_attributes = True 