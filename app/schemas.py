from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from contracts.interfaces import Driver, ForecastResult, StaticProfile, TelemetryFrame, TwinState

T = TypeVar("T")

DISCLAIMER = "Exploratory synthetic research output — not a clinical measurement or treatment recommendation."


class ApiEnvelope(BaseModel, Generic[T]):
    data: T
    model_version: str
    assumptions_version: str = "v0.1"
    is_synthetic: bool = True
    disclaimer: str = DISCLAIMER


class PatientSummary(BaseModel):
    patient_id: str
    display_name: str
    age_years: float
    ckd_stage: str
    current_risk_band: Literal["green", "amber", "red"]
    risk_48h: float
    trend: Literal["up", "down", "flat"]
    last_data_age_min: float


class TelemetryResponse(BaseModel):
    patient_id: str
    resolution_minutes: int
    window_start: datetime
    window_end: datetime
    frames: list[TelemetryFrame]


class TrajectoryPoint(BaseModel):
    timestamp: datetime
    pgc_mmhg: float
    pgc_interval: tuple[float, float]
    egfr_proxy: float
    egfr_proxy_interval: tuple[float, float]


class ForecastResponse(BaseModel):
    forecast: ForecastResult
    projected_trajectory: list[TrajectoryPoint]
    kdigo_threshold_note: str = "KDIGO staging is not inferred from this prototype trajectory."


class SimulationRequest(BaseModel):
    fluid_volume_ml: int = Field(default=0, ge=0, le=1500)
    fluid_duration_hours: float = Field(default=4.0, gt=0, le=24)
    map_target_delta_mmhg: float = Field(default=0.0, ge=-10, le=20)
    nsaid_withheld: bool = False
    acei_arb_paused: bool = False
    diuretic_paused: bool = False
    renal_vasodilator_started: bool = False
    scenario: Literal["low", "central", "high"] = "central"
    horizon_hours: int = Field(default=48, ge=6, le=48)


class SimulationResponse(BaseModel):
    baseline: ForecastResponse
    simulated: ForecastResponse
    risk_48h_range: tuple[float, float]
    state_delta: dict[str, float]
    warnings: list[str]
    assumptions_used: list[str]
    audit_id: str


class AssumptionEntry(BaseModel):
    key: str
    central: str
    low_high: str
    rationale: str
    evidence_grade: Literal["prototype", "literature-informed"]


class ModelCard(BaseModel):
    model_version: str
    trained_on: str
    limitations: list[str]
    is_stub: bool


class ErrorResponse(BaseModel):
    error: str
    message: str
    model_version: str = "unavailable"
    assumptions_version: str = "v0.1"
    is_synthetic: bool = True
    disclaimer: str = DISCLAIMER
