from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import Optional, List
from datetime import datetime

class Witness(BaseModel):
    Name: str = Field(..., max_length=25)
    Contact_No: str = Field(..., min_length=11, max_length=11)

#class CrimeType(str, Enum):
 #   car_theft = "Car Theft"
  #  robbery = "Robbery"
   # assault = "Assault"
    #motorcycle_theft = "Motorcycle Theft"
    #mobile_snatching = "Mobile Snatching"

class IntensityLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"

class CrimeCategory(BaseModel):
    Crime_Name: str = Field(..., min_length=1, max_length=50)
    Intensity_Level: IntensityLevel

class CrimeReport(BaseModel):
    Name: str = Field(..., min_length=1, max_length=25)
    Contact_No: str = Field(..., min_length=11, max_length=11)
    Crime_Type: str = Field(..., min_length=1, max_length=50)
    Intensity_Level: IntensityLevel
    Witness_Info: Optional[List[Witness]] = Field(default_factory=list)
    Location: str = Field(..., max_length=150)
    Latitude: float
    Longitude: float
    Timestamp: str
    Description: str = Field(..., max_length=100)


    @model_validator(mode="after")
    def validate_report(self) -> "CrimeReport":
        required = 0
        if self.Intensity_Level == IntensityLevel.high:
            required = 2
        elif self.Intensity_Level == IntensityLevel.critical:
            required = 4

        if len(self.Witness_Info) < required:
            raise ValueError(f"{self.Intensity_Level.value} intensity requires at least {required} witnesses.")
        return self
