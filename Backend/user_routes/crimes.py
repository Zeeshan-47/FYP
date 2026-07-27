from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
import os
from db import reports_collection, users_collection, crime_categories_collection, alerts_collection
from schemas.crime_report import CrimeReport, CrimeCategory
from functions.login_function import get_current_user, require_admin
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import itertools
from pydantic import BaseModel
import joblib
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
from dotenv import load_dotenv
import math
import requests



router = APIRouter()

new_reports_counter = 0

MODEL_FILE = "production_rf_model.joblib"

model_cache = None

def _get_shift(hour: int) -> int:
    if 0 <= hour < 6:
        return 0  # Late Night
    elif 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    else:
        return 3  # Evening/Night


def _train_model_core():
    global model_cache
    print("Random Forest model (Shift-Based)...")
    
    cursor = reports_collection.find({}, {"Location": 1, "Timestamp": 1, "Latitude": 1, "Longitude": 1})
    reports = list(cursor)

    if not reports:
        print("No reports found in database to train on.")
        return

    data = []
    loc_coords = {}
    
    for r in reports:
        loc = r.get("Location")
        ts = r.get("Timestamp")
        if not loc or not ts:
            continue
            
        dt = pd.to_datetime(ts)
        
        shift = _get_shift(dt.hour)
        data.append({"Location": loc, "Day": dt.weekday(), "Shift": shift})
        
        if loc not in loc_coords:
            loc_coords[loc] = (float(r.get('Latitude', 0)), float(r.get('Longitude', 0)))

    df = pd.DataFrame(data)

    # The Dense Grid (Locations x 7 Days x 4 Shifts)
    all_locs = df['Location'].unique()
    grid = pd.DataFrame(list(itertools.product(all_locs, range(7), range(4))), columns=['Location', 'Day', 'Shift'])
    crime_counts = df.groupby(['Location', 'Day', 'Shift']).size().reset_index(name='Crime_Count')
    training_data = pd.merge(grid, crime_counts, on=['Location', 'Day', 'Shift'], how='left').fillna(0)

    # THRESHOLD: Top 30% most dangerous shifts become Hotspots (Risk=1)
    threshold = training_data['Crime_Count'].quantile(0.70)
    training_data['Risk_Label'] = (training_data['Crime_Count'] > threshold).astype(int)

    le = LabelEncoder()
    training_data['Location_Encoded'] = le.fit_transform(training_data['Location'])

    X = training_data[['Location_Encoded', 'Day', 'Shift']]
    y = training_data['Risk_Label']

    rf = RandomForestClassifier(
        n_estimators=100,            
        max_depth=15, 
        min_samples_leaf=2,
        class_weight='balanced',     
        random_state=42
    )
    rf.fit(X, y)

    bundle = {
        "model": rf,
        "encoder": le,
        "coords": loc_coords,
        "all_locs": all_locs
    }
    joblib.dump(bundle, MODEL_FILE)
    
    model_cache = bundle
    print("AI Model trained and deployed to production!")


@router.post("/report")
def submit_report(
    report: CrimeReport, 
    background_tasks: BackgroundTasks, 
    current_user: dict = Depends(get_current_user)
):
    global new_reports_counter 
    
    if current_user["role"] != "Citizen":
        raise HTTPException(status_code=403, detail="Only registered citizens can report crime")
    
    report_dict = report.dict()
    result = reports_collection.insert_one(report_dict)
    
    new_reports_counter += 1
    
    if new_reports_counter >= 50:
        print("50 new reports received! Triggering automatic AI adaptation...")
        
        background_tasks.add_task(_train_model_core)
        
        new_reports_counter = 0

    return {"message": "Report Submitted", "id": str(result.inserted_id)}


@router.get("/reports")
def get_all_reports():    
    cursor = reports_collection.find()
    
    reports_list = []
    for doc in cursor:
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

@router.get("/my-reports")
def get_my_reports(current_user: dict = Depends(get_current_user)):

    user_contact = current_user.get("Contact_No")
    
    cursor = reports_collection.find({"Contact_No": user_contact})
    
    reports_list = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        reports_list.append(doc)
        
    return reports_list

@router.get("/admin/reports-filtered")
def get_filtered_reports(
    crime_type: Optional[str] = None,
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,  
    current_user: dict = Depends(require_admin) 
):
    query = {}

    if crime_type and crime_type != "All":
        query["Crime_Type"] = crime_type

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = f"{start_date}T00:00:00"
        if end_date:
            date_filter["$lte"] = f"{end_date}T23:59:59"
        
        if date_filter:
            query["Timestamp"] = date_filter

    cursor = reports_collection.find(query)
    
    reports_list = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        reports_list.append(doc)
        
    return reports_list

@router.get("/admin/analytics")
async def get_admin_analytics(
    crime_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user) 
):
    query = {}
    
    if crime_type and crime_type != "All":
        query["Crime_Type"] = crime_type

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = f"{start_date}T00:00:00" 
        if end_date:
            date_filter["$lte"] = f"{end_date}T23:59:59"
            
        if date_filter:
            query["Timestamp"] = date_filter

    cursor = reports_collection.find(query)
    reports = list(cursor)

    total_reports = len(reports)
    type_distribution = {}
    daily_trend = {}

    for r in reports:
        c_type = r.get("Crime_Type", "Unknown")
        type_distribution[c_type] = type_distribution.get(c_type, 0) + 1

        ts = r.get("Timestamp")
        if ts:
            try:
                date_key = ts.split("T")[0]
                daily_trend[date_key] = daily_trend.get(date_key, 0) + 1
            except Exception:
                pass

    return {
        "total_reports": total_reports,
        "type_distribution": type_distribution,
        "daily_trend": daily_trend
    }

@router.get("/admin/tactical-intelligence")
def get_tactical_intelligence(
    crime_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(require_admin)
):
    query = {}
    
    if crime_type and crime_type != "All":
        query["Crime_Type"] = crime_type

    if start_date or end_date:
        date_filter = {}
        if start_date: 
            date_filter["$gte"] = f"{start_date}T00:00:00"
        if end_date: 
            date_filter["$lte"] = f"{end_date}T23:59:59"
        if date_filter: 
            query["Timestamp"] = date_filter

    cursor = reports_collection.find(
        query, 
        {"Timestamp": 1, "Location": 1, "Crime_Type": 1, "_id": 0}
    )
    raw_data = list(cursor)

    empty_response = {
        "trend_data":[], "average_daily_crimes": 0.0, "average_trend_crimes": 0.0,
        "trend_period_type": "Daily",
        "heatmap_data": [[0 for _ in range(24)] for _ in range(7)],
        "max_heatmap_density": 1, "total_crimes_filtered": 0,
        "kpi_anomalies": 0, "kpi_high_risk_window": "N/A",
        "kpi_trend_text": "Stable", "kpi_trend_direction": "neutral",
        "kpi_top_location": "N/A"
    }

    if not raw_data:
        return empty_response

    df = pd.DataFrame(raw_data)
    df['Datetime'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Datetime'])

    if df.empty: 
        return empty_response

    df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Karachi').dt.tz_localize(None)
    total_crimes = len(df)

    kpi_top_location = "N/A"
    if 'Location' in df.columns:
        locations = df['Location'].replace("", pd.NA).dropna()
        if not locations.empty:
            top_loc_series = locations.value_counts()
            if not top_loc_series.empty:
                kpi_top_location = f"{top_loc_series.index[0]} ({int(top_loc_series.iloc[0])})"

    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour'] = df['Datetime'].dt.hour
    
    heatmap_series = df.groupby(['DayOfWeek', 'Hour']).size()
    heatmap_grid = [[0 for _ in range(24)] for _ in range(7)]
    max_heatmap_density = 0
    max_day_idx, max_hour_idx = 0, 0

    for (d, h), count in heatmap_series.items():
        heatmap_grid[d][h] = int(count)
        if count > max_heatmap_density:
            max_heatmap_density = int(count)
            max_day_idx, max_hour_idx = d, h

    timeline_start = pd.to_datetime(start_date) if start_date else df['Datetime'].min().normalize()
    timeline_end = pd.to_datetime(end_date) if end_date else df['Datetime'].max().normalize()

    if not start_date and not end_date:
        if (timeline_end - timeline_start).days < 7:
            timeline_start = timeline_end - pd.Timedelta(days=6)
            
    if timeline_start > timeline_end:
        timeline_start = timeline_end - pd.Timedelta(days=6)

    delta_days = (timeline_end - timeline_start).days
    
    if delta_days <= 62: # Up to 2 months -> DAILY
        period_type = "Daily"
        df['Period'] = df['Datetime'].dt.normalize()
        idx = pd.date_range(start=timeline_start, end=timeline_end, freq='D')
        
    elif delta_days <= 185: # Up to 6 months -> WEEKLY
        period_type = "Weekly"
        
        df['Period'] = df['Datetime'].dt.to_period('W').dt.start_time
        start_p = timeline_start.to_period('W').start_time
        end_p = timeline_end.to_period('W').start_time
        idx = pd.date_range(start=start_p, end=end_p, freq='7D')
        
    else: # Beyond 6 months -> MONTHLY
        period_type = "Monthly"
        df['Period'] = df['Datetime'].dt.to_period('M').dt.start_time
        start_p = timeline_start.to_period('M').start_time
        end_p = timeline_end.to_period('M').start_time
        idx = pd.date_range(start=start_p, end=end_p, freq='MS')

    period_counts = df.groupby('Period').size()
    period_counts.index = pd.DatetimeIndex(period_counts.index)
    period_counts = period_counts.reindex(idx, fill_value=0)

    details_map = {}
    for date_obj, group in df.groupby('Period'):
        date_str = date_obj.strftime("%Y-%m-%d")
        
        locs = group['Location'].replace("", pd.NA).dropna().value_counts().head(4)
        loc_str = " | ".join([f"{k} ({v})" for k, v in locs.items()])
        if len(group['Location'].replace("", pd.NA).dropna().unique()) > 4:
            loc_str += " | + Others"
            
        types = group['Crime_Type'].replace("", pd.NA).dropna().value_counts().head(4)
        type_str = " | ".join([f"{k} ({v})" for k, v in types.items()])
        if len(group['Crime_Type'].replace("", pd.NA).dropna().unique()) > 4:
            type_str += " | + Others"
            
        details_map[date_str] = {"locations": loc_str or "Unknown", "types": type_str or "Unknown"}

    trend_data =[]
    avg_period = float(period_counts.mean()) if not period_counts.empty else 0.0
    threshold = avg_period * 1.25
    kpi_anomalies = 0

    for date_obj, count in period_counts.items():
        date_str = date_obj.strftime("%Y-%m-%d")
        val = int(count)
        
        loc_info = details_map.get(date_str, {}).get("locations", "Clear") if val > 0 else "Clear"
        type_info = details_map.get(date_str, {}).get("types", "Clear") if val > 0 else "Clear"

        trend_data.append({
            "date": date_str, 
            "count": val,
            "locations": loc_info,
            "types": type_info
        })
        
        if val > threshold:
            kpi_anomalies += 1

    kpi_trend_text = "Stable"
    kpi_trend_dir = "neutral"
    
    if len(period_counts) >= 3:
        last_3_avg = float(period_counts.iloc[-3:].mean())
        diff = last_3_avg - avg_period
        
        # Adaptive label (e.g. +5/wk vs +12/mo)
        suffix = period_type[:2].lower()
        if diff > 0.1:
            kpi_trend_text = f"▲ +{diff:.1f}/{suffix}"
            kpi_trend_dir = "up"
        elif diff < -0.1:
            kpi_trend_text = f"▼ {abs(diff):.1f}/{suffix}"
            kpi_trend_dir = "down"

    days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    kpi_high_risk = f"{days_map[max_day_idx]} @ {max_hour_idx:02d}:00" if max_heatmap_density > 0 else "N/A"

    return {
        "trend_data": trend_data,
        "average_trend_crimes": round(avg_period, 2), 
        "trend_period_type": period_type,             
        "heatmap_data": heatmap_grid,
        "max_heatmap_density": max_heatmap_density,
        "total_crimes_filtered": total_crimes,
        "kpi_anomalies": kpi_anomalies,
        "kpi_high_risk_window": kpi_high_risk,
        "kpi_trend_text": kpi_trend_text,
        "kpi_trend_direction": kpi_trend_dir,
        "kpi_top_location": kpi_top_location
    }

@router.get("/analytics/map-data")
async def get_map_data(
    crime_type: str = Query("All"),
    intensity: str = Query("All"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    try:
        # 1. Build Strict MongoDB Query
        query = {}
        if crime_type != "All": 
            query["Crime_Type"] = crime_type
        if intensity != "All": 
            query["Intensity_Level"] = intensity
            
        # (Optional) Time-based filtering if you pass dates from Flutter
        if start_date or end_date:
            query["Timestamp"] = {}
            if start_date: query["Timestamp"]["$gte"] = start_date
            if end_date: query["Timestamp"]["$lte"] = end_date
            
        projection = {
            "Latitude": 1, "Longitude": 1, 
            "Crime_Type": 1, "Intensity_Level": 1, 
            "Timestamp": 1, "Location": 1, "Description": 1
        }
        
        # 2. Fetch from MongoDB (Async)
        cursor = reports_collection.find(query, projection)
        reports = cursor.to_list(length=None) 

        if not reports:
            return []
        
        # 3. ENTERPRISE OPTIMIZATION: Use Pandas for Vectorized Processing
        # This is infinitely faster than Python 'for' loops for spatial data
        df = pd.DataFrame(reports)
        
        # Safely force coordinates to floats. Any corrupt text/data becomes NaN
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Drop rows where coordinates are NaN, 0, or geographically impossible
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df = df[
            (df['Latitude'] >= -90) & (df['Latitude'] <= 90) & 
            (df['Longitude'] >= -180) & (df['Longitude'] <= 180) &
            (df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)
        ]
        
        if df.empty:
            return []
            
        # 4. Extract clean coordinates as a high-speed Numpy array
        coords = df[['Latitude', 'Longitude']].to_numpy()
        total_points = len(coords)
        
        # 5. DYNAMIC DENSITY SCALING
        # Adjust minimum cluster size based on how much data is on screen.
        # If showing thousands of points, require 10 to form a cluster.
        # If showing a highly filtered view, require 3.
        if total_points > 500:
            min_samples = 10
        elif total_points > 100:
            min_samples = 5
        else:
            min_samples = 3
            
        # 6. Background Machine Learning (DBSCAN)
        if total_points >= min_samples and min_samples > 1:
            # Run heavy ML matrix math on a separate thread to prevent API blocking
            cluster_labels = await asyncio.to_thread(_run_dbscan, coords, min_samples)
            df['cluster_id'] = cluster_labels
        else:
            # Not enough data for AI clustering, treat all as standalone pins (-1)
            df['cluster_id'] = -1
            
        # 7. Safe Serialization for Flutter
        # Convert MongoDB ObjectIds to strings
        df['_id'] = df['_id'].astype(str)
        
        # Replace any remaining NaNs (like missing Descriptions) with empty strings
        df = df.fillna("")
        
        # Convert DataFrame back to a list of dicts for JSON return
        return df.to_dict(orient='records')

    except Exception as e:
        print(f"[CRITICAL ERROR] Map Data Processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during spatial clustering.")

def _run_dbscan(coords: np.ndarray, min_samples: int) -> list:
    """
    Executes Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Utilizes the Haversine formula for perfectly accurate spherical Earth distances.
    """
    kms_per_radian = 6371.0088
    
    # Radius threshold: 0.4km (400 meters). Perfect for neighborhood block-level accuracy.
    eps_in_radians = 0.4 / kms_per_radian 
    
    db = DBSCAN(
        eps=eps_in_radians, 
        min_samples=min_samples, 
        metric='haversine', 
        algorithm='ball_tree'
    ).fit(np.radians(coords))
    
    return db.labels_.tolist()

@router.get("/analytics/dashboard-data2")
async def get_dashboard_data2():
    try:
        reports = list(reports_collection.find({}))
        for r in reports:
            r["_id"] = str(r["_id"])
        return reports

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/dashboard-data")
async def get_dashboard_data(
    crime_type: str = Query("All", description="Filter by crime type"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format"),
    end_date: Optional[str] = Query(None, description="End date in ISO format"),
    comparison: str = Query("First vs Second Half", description="Comparison period type")
):
    try:
        query = {}
        if crime_type != "All":
            query["Crime_Type"] = crime_type
            
        if start_date and end_date:
            query["Timestamp"] = {"$gte": start_date, "$lte": end_date}

        reports = list(reports_collection.find(query, {"Timestamp": 1, "_id": 0}))
        
        monthly_counts = {i: 0 for i in range(1, 13)}
        h1_count = 0
        h2_count = 0
        comp_h1 = 0
        comp_h2 = 0
        now = datetime.utcnow()

        for r in reports:
            ts_str = r.get("Timestamp")
            if not ts_str:
                continue
                
            try:
                d = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                continue
                
            monthly_counts[d.month] += 1
            
            if d.month <= 6:
                h1_count += 1
            else:
                h2_count += 1
                
            if comparison == "First vs Second Half":
                if d.month <= 6: comp_h1 += 1
                else: comp_h2 += 1
            elif comparison == "Quarterly (Q1 vs Q2)":
                if 1 <= d.month <= 3: comp_h1 += 1
                elif 4 <= d.month <= 6: comp_h2 += 1
            elif comparison == "Last 28 Days":
                days_diff = (now.replace(tzinfo=d.tzinfo) - d).days
                if 0 <= days_diff <= 28:
                    comp_h2 += 1 # Current 28 Days
                elif 28 < days_diff <= 56:
                    comp_h1 += 1 # Previous 28 Days

        peak_month_num = max(monthly_counts, key=monthly_counts.get)
        peak_month_val = monthly_counts[peak_month_num]
        months =['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        peak_month_label = months[peak_month_num - 1] if peak_month_val > 0 else "N/A"
        
        # Rate of change formula
        global_change = ((h2_count - h1_count) / h1_count * 100) if h1_count > 0 else (100.0 if h2_count > 0 else 0.0)

        return {
            "total_incidents": len(reports),
            "peak_month": peak_month_label,
            "monthly_trend": [{"month": k, "count": v} for k, v in monthly_counts.items()],
            "global_metrics": {
                "h1": h1_count,
                "h2": h2_count,
                "change": global_change,
                "is_decreasing": h2_count <= h1_count
            },
            "comparison_metrics": {
                "period_1_count": comp_h1,
                "period_2_count": comp_h2
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/admin/dashboard-filters")
def get_dashboard_filters(current_user: dict = Depends(require_admin)):
    locations = reports_collection.distinct("Location")
    crime_types = reports_collection.distinct("Crime_Type")
    
    return {
        "locations": sorted([l for l in locations if l and isinstance(l, str)]),
        "crime_types": sorted([c for c in crime_types if c and isinstance(c, str)])
    }

@router.get("/admin/dashboard-data")
def get_dashboard_data3(
    area: Optional[str] = "Total Area", 
    category: Optional[str] = "All",
    current_user: dict = Depends(require_admin)
):
    query = {}
    
    if area and area != "Total Area":
        query["Location"] = area
    if category and category != "All":
        query["Crime_Type"] = category
        
    projection = {
        "Timestamp": 1, 
        "Location": 1, 
        "Crime_Type": 1, 
        "Latitude": 1, 
        "Longitude": 1
    }
    
    cursor = reports_collection.find(query, projection)
    
    data = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        data.append(doc)
        
    return data

@router.post("/admin/crime-category")
def add_crime_category(category: CrimeCategory):
    existing = crime_categories_collection.find_one(
        {"Crime_Name": {"$regex": f"^{category.Crime_Name}$", "$options": "i"}}
    )
    if existing:
        raise HTTPException(status_code=400, detail="This crime type already exists.")
    
    new_category = {
        "Crime_Name": category.Crime_Name,
        "Intensity_Level": category.Intensity_Level.value
    }
    
    crime_categories_collection.insert_one(new_category)
    return {"message": "Crime category added successfully!"}

@router.delete("/admin/crime-category/{crime_name}")
def delete_crime_category(crime_name: str):
    result = crime_categories_collection.delete_one({"Crime_Name": crime_name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Crime not found.")
    return {"message": "Crime deleted successfully!"}

@router.get("/crime-categories")
def get_crime_categories():
    cursor = crime_categories_collection.find({}, {"_id": 0})
    return list(cursor)

def load_model_to_ram():
    global model_cache
    if os.path.exists(MODEL_FILE):
        model_cache = joblib.load(MODEL_FILE)
        print("Production Model loaded into RAM.")
    else:
        print("No model found. Training initial model...")
        _train_model_core()

load_model_to_ram()


@router.get("/analytics/predict-risk")
async def predict_risk(day_of_week: int = Query(...), hour: int = Query(...)):
    global model_cache
    
    if model_cache is None:
        return []

    try:
        rf = model_cache["model"]
        le = model_cache["encoder"]
        loc_coords = model_cache["coords"]
        all_locs = model_cache["all_locs"]

        # TRANSLATE Flutter's requested Hour into the AI's expected Shift
        target_shift = _get_shift(hour)

        pred_df = pd.DataFrame({'Location': all_locs, 'Day': day_of_week, 'Shift': target_shift})
        pred_df['Location_Encoded'] = le.transform(pred_df['Location'])

        probabilities = rf.predict_proba(pred_df[['Location_Encoded', 'Day', 'Shift']])[:, 1]
        
        results = []
        for idx, row in pred_df.iterrows():
            loc_name = row['Location']
            if loc_name in loc_coords:
                lat, lng = loc_coords[loc_name]
                results.append({
                    "Sector": loc_name,
                    "Latitude": lat,
                    "Longitude": lng,
                    "Risk_Percentage": round(probabilities[idx] * 100, 1)
                })

        return sorted(results, key=lambda x: x['Risk_Percentage'], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/active-alerts")
async def get_active_alerts():
    try:
        alerts = await asyncio.to_thread(_detect_anomalies)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _detect_anomalies():
    try:
        cursor = reports_collection.find({}, {"Location": 1, "Timestamp": 1})
        reports = list(cursor)
        
        if not reports:
            # Return any existing active alerts in DB even if no reports exist
            return _fetch_active_alerts_from_db()
            
        df = pd.DataFrame(reports)
        df = df.dropna(subset=['Timestamp', 'Location'])
        df['Location'] = df['Location'].astype(str).str.lower().str.replace(' area', '').str.strip().str.title()
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")
        df = df.dropna(subset=['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date
        
        daily_counts = df.groupby(['Location', 'Date']).size().reset_index(name='Daily_Count')
        
        for loc in daily_counts['Location'].unique():
            loc_data = daily_counts[daily_counts['Location'] == loc].copy()
            loc_data = loc_data.sort_values('Date')
            
            if len(loc_data) < 14:
                continue
                
            loc_data['Day_of_Week'] = pd.to_datetime(loc_data['Date']).dt.dayofweek
            features = loc_data[['Daily_Count', 'Day_of_Week']]
            
            model = IsolationForest(contamination=0.05, random_state=42)
            loc_data['Anomaly'] = model.fit_predict(features)
            
            normal_days = loc_data[loc_data['Anomaly'] == 1]
            normal_mean = normal_days['Daily_Count'].mean()
            
            if pd.isna(normal_mean) or normal_mean <= 0:
                normal_mean = 1.0 
            
            latest_record = loc_data.iloc[-1]
            
            # --- ANOMALY DETECTED ---
            if latest_record['Anomaly'] == -1 and latest_record['Daily_Count'] > (normal_mean * 1.5):
                spike_pct = int(((latest_record['Daily_Count'] - normal_mean) / normal_mean) * 100)
                alert_date_str = str(latest_record['Date'])

                alert_doc = {
                    "Sector": loc,
                    "Date": alert_date_str,
                    "Current_Count": int(latest_record['Daily_Count']),
                    "Normal_Baseline": round(normal_mean, 1),
                    "Spike_Percentage": spike_pct,
                    "Decision": f"Issue a 'High Alert' notification to app users in the {loc} area.",
                    "status": "active",
                    "createdAt": datetime.utcnow()
                }

                # 1. UPSERT TO MONGODB:
                # Insert ONLY if an alert for this Sector & Date does not exist yet.
                # $setOnInsert prevents overwriting status if an admin already marked it 'resolved'.
                alerts_collection.update_one(
                    {"Sector": loc, "Date": alert_date_str},
                    {"$setOnInsert": alert_doc},
                    upsert=True
                )

        # 2. FETCH ALL ACTIVE ALERTS FROM MONGODB (WITH _id)
        return _fetch_active_alerts_from_db()

    except Exception as e:
        raise e


# Helper function to fetch active alerts from MongoDB
def _fetch_active_alerts_from_db():
    cursor = alerts_collection.find({"status": "active"}).sort("createdAt", -1)
    active_alerts = list(cursor)
    
    # Format MongoDB ObjectId (_id) to string for Flutter JSON parsing
    for alert in active_alerts:
        alert["_id"] = str(alert["_id"])
        
    active_alerts.sort(key=lambda x: x.get("Spike_Percentage", 0), reverse=True)
    return active_alerts


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=GenerationConfig(
        response_mime_type="application/json"
    )
)

class FIRRequest(BaseModel):
    description: str
    crime_type: str = "Unknown"

@router.post("/analytics/analyze-fir")
async def analyze_fir(request: FIRRequest):
    try:
        result = await asyncio.to_thread(_run_gemini_summarizer, request.description, request.crime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _run_gemini_summarizer(description: str, crime_type: str):
    # This prompt is the "brain" of your new architecture
    prompt = f"""
    You are an expert police dispatcher assistant in Quetta, Pakistan. 
    Analyze the following citizen FIR report.
    
    Reported Category: {crime_type}
    Citizen's Description: {description}

    Extract the most important information and respond ONLY in the following JSON schema:
    {{
        "Summary": "A concise, professional 1-2 sentence summary of what happened for the police officer to read quickly.",
        "Extracted_Entities": ["list", "of", "weapons", "vehicles", "suspect descriptions", "locations mentioned"],
        "Calculated_Priority": "CRITICAL (if weapons, violence, or high threat) or STANDARD",
        "Actionable_Decision": "1 specific recommended action for the police unit.",
        "Color_Code": "RED if CRITICAL, GREEN if STANDARD"
    }}
    """
    
    response = model.generate_content(prompt)

    text = response.text.replace("```json", "").replace("```", "").strip()
    
    ai_data = json.loads(text)
    
    return {
        "Original_Text": description,
        "Top_Intent": crime_type.upper(),
        "Summary": ai_data.get("Summary", "No summary generated."),
        "Extracted_Entities": ai_data.get("Extracted_Entities", []),
        "Calculated_Priority": ai_data.get("Calculated_Priority", "STANDARD"),
        "Actionable_Decision": ai_data.get("Actionable_Decision", "Review manually."),
        "Color_Code": ai_data.get("Color_Code", "GREEN")
    }


@router.get("/analytics/recent-reports")
async def get_recent_reports():
    try:
        cursor = reports_collection.find(
            {}, 
            {"Name": 1, "Crime_Type": 1, "Timestamp": 1, "Description": 1, "Location": 1}
        ).sort([("Timestamp", -1)]).limit(20)
        
        reports = list(cursor)
        
        for r in reports:
            r["_id"] = str(r["_id"])
            
        return reports
        
    except Exception as e:
        print(f"Error fetching reports: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/analytics/user-count")
async def get_user_count():
    try:
        count = users_collection.count_documents({})
        
        return {"total_users": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_recent_crimes_from_db():
    try:
        # 1. Query: Only get reports that actually have GPS coordinates
        query = {
            "Latitude": {"$exists": True, "$ne": "", "$ne": None},
            "Longitude": {"$exists": True, "$ne": "", "$ne": None}
        }
        
        # 2. Projection: We ONLY fetch what the math engine needs to save RAM
        projection = {
            "Crime_Type": 1, 
            "Intensity_Level": 1, 
            "Latitude": 1, 
            "Longitude": 1
        }
        
        # 3. Fetch the latest 500 crimes to create a dense, accurate risk map
        cursor = reports_collection.find(query, projection).sort([("Timestamp", -1)]).limit(500)
        
        crimes = list(cursor)
        
        # Safely convert ObjectIds and coordinates
        for c in crimes:
            c["_id"] = str(c["_id"])
            # Ensure coordinates are floats so math.sin() doesn't crash
            c["Latitude"] = float(c.get("Latitude", 0))
            c["Longitude"] = float(c.get("Longitude", 0))
            
        return crimes
        
    except Exception as e:
        print(f"Error fetching crimes for routing: {e}") 
        return [] # Return empty list so the routing doesn't crash, it will just assume 0 risk

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"

class RouteRequest(BaseModel):
    source: str
    destination: str
    mode: str = "citizen" # 'citizen' or 'patrol'

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def geocode_location(location_name: str):
    # Append Quetta to improve accuracy
    query = f"{location_name}, Quetta, Pakistan"
    try:
        resp = requests.get(NOMINATIM_URL, params={"q": query, "format": "json", "limit": 1}, headers={"User-Agent": "CrimeApp/1.0"}, timeout=10)
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    return None, None

def get_routes_from_osrm(src_lat, src_lon, dst_lat, dst_lon):
    try:
        url = f"{OSRM_URL}/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        resp = requests.get(url, params={"overview": "full", "geometries": "geojson", "alternatives": "true", "steps": "false"}, timeout=15)
        data = resp.json()
        routes = []
        if data.get("code") == "Ok":
            for i, route in enumerate(data.get("routes", [])):
                coords = [(c[1], c[0]) for c in route["geometry"]["coordinates"]] # Swap to Lat, Lon
                routes.append({
                    "index": i,
                    "coords": coords,
                    "distance_km": round(route["distance"] / 1000, 2),
                    "duration_min": round(route["duration"] / 60, 1)
                })
        return routes
    except:
        return []

def calculate_route_risk(route_coords, crimes, radius_km=0.4):
    sample_coords = route_coords[::10] if len(route_coords) > 20 else route_coords
    nearby_crimes = set()
    high_priority_near = 0

    for r_lat, r_lon in sample_coords:
        for crime in crimes:
            c_id = str(crime.get('_id', ''))
            c_lat = float(crime.get('Latitude', 0))
            c_lon = float(crime.get('Longitude', 0))
            
            if c_lat == 0 or c_lon == 0: continue

            dist = haversine_distance(r_lat, r_lon, c_lat, c_lon)
            if dist <= radius_km and c_id not in nearby_crimes:
                nearby_crimes.add(c_id)
                if crime.get('Intensity_Level') in ['High', 'Critical'] or crime.get('Crime_Type') in ['Assault', 'Robbery', 'Murder']:
                    high_priority_near += 1

    crime_count = len(nearby_crimes)
    raw_score = (crime_count * 2.5) + (high_priority_near * 6.0)
    risk_score = int(100 * (1 - math.exp(-raw_score / 50.0)))
    return max(0, min(risk_score, 100)), crime_count, high_priority_near

@router.post("/analytics/plan-route")
async def plan_route(request: RouteRequest):
    def process():
        src_lat, src_lon = geocode_location(request.source)
        dst_lat, dst_lon = geocode_location(request.destination)
        
        if not src_lat or not dst_lat:
            raise ValueError("Could not find coordinates for the given locations.")

        osrm_routes = get_routes_from_osrm(src_lat, src_lon, dst_lat, dst_lon)
        crimes = get_recent_crimes_from_db() # Call your DB here

        results = []
        for r in osrm_routes:
            score, c_count, hp_count = calculate_route_risk(r['coords'], crimes)
            
            if request.mode == "patrol":
                if score >= 70: classification = {"level": "Optimal Patrol", "color": "#3b82f6"} # Blue
                elif score >= 40: classification = {"level": "Standard Patrol", "color": "#f59e0b"} # Yellow
                else: classification = {"level": "Low Utility", "color": "#10b981"} # Green (Safe, so low patrol utility)
                
                prompt = f"Analyze patrol route: {request.source} to {request.destination}. Score: {score}/100. Crimes: {c_count}. High priority: {hp_count}. Explain why it is {classification['level']} in 2 sentences."
            else:
                if score <= 30: classification = {"level": "Safe Route", "color": "#10b981"} # Green
                elif score <= 60: classification = {"level": "Moderate Risk", "color": "#f59e0b"} # Yellow
                else: classification = {"level": "High Risk - Avoid", "color": "#f43f5e"} # Red
                
                prompt = f"Analyze travel route: {request.source} to {request.destination}. Score: {score}/100. Crimes: {c_count}. High priority: {hp_count}. Explain why it is {classification['level']} in 2 sentences."

            try:
                ai_text = model.generate_content(prompt).text.strip()
            except:
                ai_text = "AI explanation unavailable at the moment."

            results.append({
                "id": r["index"],
                "distance_km": r["distance_km"],
                "duration_min": r["duration_min"],
                "score": score,
                "classification": classification["level"],
                "color": classification["color"],
                "crime_count": c_count,
                "high_priority_count": hp_count,
                "ai_explanation": ai_text,
                "coordinates": r["coords"]
            })
            
        return {
            "source_coords": [src_lat, src_lon],
            "dest_coords": [dst_lat, dst_lon],
            "routes": results
        }

    try:
        return await asyncio.to_thread(process)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))