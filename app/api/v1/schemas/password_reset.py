# app/api/v1/schemas/password_reset.py
from pydantic import BaseModel, EmailStr, Field

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetVerify(BaseModel):
    email: EmailStr
    code: str

class PasswordResetConfirm(BaseModel):
    email: EmailStr
    code: str
    new_password: str = Field(min_length=8)
    confirm_password: str = Field(min_length=8)
