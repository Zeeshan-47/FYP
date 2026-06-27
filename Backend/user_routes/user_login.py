from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from functions.login_function import authenticate_user, create_access_token, get_current_user
from datetime import timedelta

router = APIRouter()

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

    token = create_access_token({"sub": user["Name"], "role": user["role"]}, timedelta(minutes=60))
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


@router.get("/users/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    
    current_user["id"] = str(current_user["_id"])
    del current_user["_id"]
    
    if "Password" in current_user:
        del current_user["Password"]
        
    return current_user