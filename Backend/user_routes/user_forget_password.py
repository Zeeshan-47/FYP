from fastapi import APIRouter, HTTPException
from schemas.user_model import PasswordResetRequest, PasswordResetConfirm
from db import users_collection
from functions.registration_function import hash_password, generate_otp, send_email
from datetime import datetime, timedelta

store_otp = {}
router = APIRouter()

@router.post("/forgot-password")
def forgot_password(request: PasswordResetRequest):
    user = users_collection.find_one({"Email": request.Email})
    if not user:
        return {"message": "If email exists, OTP sent."}

    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=5)
    store_otp[request.Email] = (otp, expiry)
    
    try:
        send_email(request.Email, otp)
        return {"message": "OTP Sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to send email")


@router.post("/reset-password")
def reset_password_confirm(data: PasswordResetConfirm):
    stored = store_otp.get(data.Email)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid or Expired OTP")
    
    otp, expiry = stored
    
    if datetime.utcnow() > expiry:
        del store_otp[data.Email]
        raise HTTPException(status_code=400, detail="OTP Expired")
    
    if otp != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    new_hashed = hash_password(data.new_password)
    users_collection.update_one(
        {"Email": data.Email},
        {"$set": {"Password": new_hashed}}
    )
    del store_otp[data.Email]
    return {"message": "Password updated successfully"}