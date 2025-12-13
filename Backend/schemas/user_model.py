from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    Name: str = Field(..., max_length=25)
    Email: EmailStr
    Contact_No: str = Field(..., min_length=11, max_length=11)
    CNIC: str = Field(..., min_length=13, max_length=13)
    Password: str = Field(..., max_length=8)
    otp: str
    role: str = "Citizen"


class OTP(BaseModel):
    Email: EmailStr


class PasswordResetRequest(BaseModel):
    Email: EmailStr


class PasswordResetConfirm(BaseModel):
    Email: EmailStr
    otp: str
    new_password: str = Field(..., max_length=8)