from fastapi import FastAPI
from user_routes.crimes import router as crime
from user_routes.user_registration import router as register
from user_routes.user_login import router as login
from user_routes.user_forget_password import router as forgot_password

app = FastAPI()
app.include_router(crime, prefix="/crime")
app.include_router(register, prefix="/crime")
app.include_router(login, prefix="/crime")
app.include_router(forgot_password, prefix="/crime")
