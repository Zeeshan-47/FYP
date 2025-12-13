from fastapi import APIRouter, Depends, HTTPException
from db import reports_collection, users_collection
from schemas.crime_report import CrimeReport
from functions.login_function import get_current_user, require_admin

router = APIRouter()


@router.post("/report")
def submit_report(report: CrimeReport, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "Citizen":
        raise HTTPException(status_code=403, detail="Only registered citizens can report crime")
    
    report_dict = report.dict()
    result = reports_collection.insert_one(report_dict)
    return {"message": "Report Submitted", "id": str(result.inserted_id)}


@router.get("/reports")
def get_all_reports():
    # 1. Fetch all documents
    cursor = reports_collection.find()
    
    reports_list = []
    for doc in cursor:
        # 2. FIX: Convert ObjectId to string to prevent crash
        doc["id"] = str(doc["_id"])
        del doc["_id"] 
        reports_list.append(doc)
        
    return reports_list

@router.post("/delete_User")
def delete_user(username: str, admin: str = Depends(require_admin)):
    result = users_collection.users.delete_one({"username": username})
    if result.deleted_count == 1:
        return {"message": f"User '{username}' deleted"}
    raise HTTPException(status_code=404, detail="User not found")