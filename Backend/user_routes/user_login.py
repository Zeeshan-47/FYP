from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from functions.login_function import authenticate_user, create_access_token
from datetime import timedelta

router = APIRouter()

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    token = create_access_token({"sub": user["Name"], "role": user["role"]}, timedelta(minutes=60))
    return {"access_token": token, "token_type": "bearer"}
