from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
from api.calculations import calculate_clogging
import joblib
import pandas as pd

app = FastAPI()

# Load models and limits at startup for validation
_MODELS_PATH = "api/all_models.pkl"
_LIMITS_PATH = "data/asset_limits.csv"
_all_models = joblib.load(_MODELS_PATH)
_limits = pd.read_csv(_LIMITS_PATH, dtype={"asset": str})
_limits["asset"] = _limits["asset"].astype(str)
_limits.set_index("asset", inplace=True)

class EstimateClogRequest(BaseModel):
    asset_type: str = Field(..., description="Asset type identifier")
    horsepower: float = Field(..., gt=0, description="Current hydraulic horsepower reading")
    measured_restriction: float = Field(..., ge=0, description="Current air filter restriction reading")

    @validator("asset_type")
    def asset_type_not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("asset_type must be a non-empty string")
        return v

class EstimateClogResponse(BaseModel):
    hp_max_current: float
    percent_clogged: float

from api.calculations import _predict_restriction_from_hp

class PredictCleanRestrictionRequest(BaseModel):
    asset_type: str = Field(..., description="Asset type identifier")
    horsepower: float = Field(..., gt=0, description="Hydraulic horsepower")

    @validator("asset_type")
    def asset_type_not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("asset_type must be a non-empty string")
        return v

class PredictCleanRestrictionResponse(BaseModel):
    predicted_restriction: float

@app.post("/predict_clean_restriction", response_model=PredictCleanRestrictionResponse)
def predict_clean_restriction(data: PredictCleanRestrictionRequest):
    asset_type = str(data.asset_type)
    if asset_type not in _all_models:
        raise HTTPException(status_code=400, detail=f"Asset type '{asset_type}' not found in models.")
    if asset_type not in _limits.index:
        raise HTTPException(status_code=400, detail=f"Asset type '{asset_type}' not found in limits.")

    max_hp = float(_limits.loc[asset_type, "Max_Horsepower"])
    if not (0 < data.horsepower <= max_hp):
        raise HTTPException(
            status_code=400,
            detail=f"Horsepower must be between 0 and {max_hp} for asset type '{asset_type}'."
        )

    try:
        restriction = _predict_restriction_from_hp(asset_type, data.horsepower)
        return PredictCleanRestrictionResponse(predicted_restriction=round(float(restriction), 4))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
@app.post("/estimate_clog", response_model=EstimateClogResponse)
def estimate_clog(data: EstimateClogRequest):
    asset_type = str(data.asset_type)
    # Validate asset_type exists
    if asset_type not in _all_models:
        raise HTTPException(status_code=400, detail=f"Asset type '{asset_type}' not found in models.")
    if asset_type not in _limits.index:
        raise HTTPException(status_code=400, detail=f"Asset type '{asset_type}' not found in limits.")

    # Get limits for this asset
    max_hp = float(_limits.loc[asset_type, "Max_Horsepower"])
    max_restriction = float(_limits.loc[asset_type, "Max_AirFilterRestriction"])

    # Validate horsepower and restriction
    if not (0 <= data.horsepower <= max_hp):
        raise HTTPException(
            status_code=400,
            detail=f"Horsepower must be between 0 and {max_hp} for asset type '{asset_type}'."
        )
    if not (0 <= data.measured_restriction <= max_restriction):
        raise HTTPException(
            status_code=400,
            detail=f"Restriction must be between 0 and {max_restriction} for asset type '{asset_type}'."
        )

    try:
        result = calculate_clogging(
            asset_type=asset_type,
            hp=data.horsepower,
            restriction=data.measured_restriction
        )
        return EstimateClogResponse(
            hp_max_current=result["HP_max_current"],
            percent_clogged=round(result["percent_clogged"], 2)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")