from fastapi import APIRouter, HTTPException
from schemas.user_model import User, OTP
from db import users_collection
from functions.registration_function import hash_password, generate_otp, send_email
from datetime import datetime, timedelta


store_otp = {}

router = APIRouter()


@router.post("/request-otp")
def request_OTP(request: OTP):

    if users_collection.find_one({"Email": request.Email}):
        raise HTTPException(status_code=400, detail="Email already registered. Please login.")
    
    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=2)
    store_otp[request.Email] = (otp, expiry)
    send_email(request.Email, otp)
    return {"Message": "OTP SEND"}


@router.post("/register")
def register_user(user: User):
    if users_collection.find_one({"Email": user.Email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    if users_collection.find_one({"CNIC": user.CNIC}):
        raise HTTPException(status_code=400, detail="CNIC already registered")
    
    if users_collection.find_one({"Contact_No": user.Contact_No}):
        raise HTTPException(status_code=400, detail="Contact_No already registered")
    
    stored = store_otp.get(user.Email)
    if not stored:
        raise HTTPException(status_code=400, detail="NO OTP FOUND")
    
    otp, expiry = stored

    if datetime.utcnow() > expiry:
        del store_otp[user.Email]
        raise HTTPException(status_code=400, detail="OTP Expired")
    
    if otp != user.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    user_dict = user.dict(exclude={"otp"})
    user_dict["Password"] = hash_password(user.Password)
    result = users_collection.insert_one(user_dict)
    
    del store_otp[user.Email]
    return {"Message": "User Registered", "id": str(result.inserted_id)}