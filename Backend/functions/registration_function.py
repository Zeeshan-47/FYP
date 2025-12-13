from email.message import EmailMessage
from passlib.context import CryptContext
import random
import smtplib

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str):
    return pwd.hash(password)


def generate_otp():
    return str(random.randint(100000, 999999))


def send_email(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f"YOUR OTP IS {otp}")
    msg["Subject"] = "Email Verification"
    msg["From"] = "raspberrypicourse5@gmail.com"
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login('raspberrypicourse5@gmail.com', 'ecww veph upin aarz')
        smtp.send_message(msg)