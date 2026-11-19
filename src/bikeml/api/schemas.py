"""Schémas d'entrée/sortie de l'API.

La validation est stricte : types, bornes et champs inconnus interdits. Une
requête non conforme est refusée avec un code 422 et un message explicite,
avant d'atteindre le modèle.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BikeFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "season": 3, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1,
        "weathersit": 1, "temp": 0.72, "atemp": 0.66, "hum": 0.45, "windspeed": 0.19}})

    season: int = Field(..., ge=1, le=4, description="1 hiver, 2 printemps, 3 été, 4 automne")
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4, description="1 clair ... 4 très dégradé")
    temp: float = Field(..., ge=0, le=1, description="température normalisée (/41 °C)")
    atemp: float = Field(..., ge=0, le=1, description="ressentie normalisée (/50 °C)")
    hum: float = Field(..., ge=0, le=1)
    windspeed: float = Field(..., ge=0, le=1)


class BatchRequest(BaseModel):
    items: list[BikeFeatures] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    request_id: str
    prediction: float
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    request_id: str
    predictions: list[float]
    model_version: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    item_index: int = Field(0, ge=0)
    actual_cnt: float = Field(..., ge=0)
