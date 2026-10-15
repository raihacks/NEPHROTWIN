from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.demo import DEMO_PATIENTS
from app.schemas import ApiEnvelope, ModelCard, PatientSummary, SimulationRequest, SimulationResponse, TelemetryResponse
from app.services import ASSUMPTIONS, ASSUMPTIONS_VERSION, ForecastService, audit_id, make_perturbation, simulation_warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecasts = ForecastService()
    yield


app = FastAPI(title="NephroTwin BE2 API", version="0.1.0", lifespan=lifespan)


def service() -> ForecastService:
    return app.state.forecasts


def envelope(data):
    return ApiEnvelope(data=data, model_version=service().model_version, assumptions_version=ASSUMPTIONS_VERSION)


def patient_or_404(patient_id: str):
    if patient_id not in DEMO_PATIENTS:
        raise HTTPException(status_code=404, detail="Unknown pseudonymous patient ID.")
    return DEMO_PATIENTS[patient_id]


@app.get("/api/v1/health")
def health():
    return {"status": "ok" if service().forecaster else "degraded", "model_version": service().model_version, "assumptions_version": ASSUMPTIONS_VERSION, "is_synthetic": True, "model_load_error": service().load_error}


@app.get("/api/v1/patients", response_model=ApiEnvelope[list[PatientSummary]])
def patients():
    summaries = []
    for patient_id, patient in DEMO_PATIENTS.items():
        try:
            forecast, _ = service().forecast(patient_id)
            risk, band = forecast.forecast.risk_48h, forecast.forecast.risk_band
        except RuntimeError:
            risk, band = 0.0, "green"
        summaries.append(PatientSummary(patient_id=patient_id, display_name=patient["display_name"], age_years=patient["profile"].age_years, ckd_stage=patient["ckd_stage"], current_risk_band=band, risk_48h=risk, trend=patient["trend"], last_data_age_min=0.0))
    return envelope(summaries)


@app.get("/api/v1/patients/{patient_id}/profile")
def profile(patient_id: str):
    return envelope(patient_or_404(patient_id)["profile"])


@app.get("/api/v1/patients/{patient_id}/telemetry")
def telemetry(patient_id: str, hours: int = Query(default=48, ge=1, le=48)):
    patient = patient_or_404(patient_id)
    frames = patient["telemetry"][-hours:]
    return envelope(TelemetryResponse(patient_id=patient_id, resolution_minutes=60, window_start=frames[0].timestamp, window_end=frames[-1].timestamp, frames=frames))


@app.get("/api/v1/patients/{patient_id}/twin-state")
def twin_state(patient_id: str):
    patient_or_404(patient_id)
    try:
        _, state = service().forecast(patient_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return envelope(state)


@app.get("/api/v1/patients/{patient_id}/forecast")
def forecast(patient_id: str):
    patient_or_404(patient_id)
    try:
        response, _ = service().forecast(patient_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return envelope(response)


@app.get("/api/v1/patients/{patient_id}/drivers")
def drivers(patient_id: str):
    patient_or_404(patient_id)
    response, _ = service().forecast(patient_id)
    return envelope(response.forecast.drivers)


@app.post("/api/v1/patients/{patient_id}/simulate")
def simulate(patient_id: str, request: SimulationRequest):
    patient_or_404(patient_id)
    try:
        baseline, base_state = service().forecast(patient_id)
        simulated, simulated_state = service().forecast(patient_id, make_perturbation(request), request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scenario_results = []
    for scenario in ("low", "high"):
        scenario_request = request.model_copy(update={"scenario": scenario})
        scenario_result, _ = service().forecast(patient_id, make_perturbation(scenario_request), scenario_request)
        scenario_results.append(scenario_result.forecast.risk_48h)
    response = SimulationResponse(baseline=baseline, simulated=simulated, risk_48h_range=(min(scenario_results), max(scenario_results)), state_delta={"ra": round(simulated_state.ra.value - base_state.ra.value, 3), "re": round(simulated_state.re.value - base_state.re.value, 3), "pgc_mmhg": round(simulated_state.pgc.value - base_state.pgc.value, 3)}, warnings=simulation_warnings(patient_id, request), assumptions_used=[entry.key for entry in ASSUMPTIONS], audit_id=audit_id())
    return envelope(response)


@app.get("/api/v1/assumptions")
def assumptions():
    return envelope(ASSUMPTIONS)


@app.get("/api/v1/model-card")
def model_card():
    forecaster = service().forecaster
    card = ModelCard(model_version=service().model_version, trained_on=forecaster.meta.trained_on if forecaster else "unavailable", is_stub=forecaster.meta.is_stub if forecaster else True, limitations=["Synthetic demo patients only.", "Glomerular values are model-estimated.", "Not validated for clinical decision-making."])
    return envelope(card)
