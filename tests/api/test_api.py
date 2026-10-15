from app.main import app, forecast, health, patients, simulate
from app.schemas import SimulationRequest
from app.services import ForecastService


def setup_module():
    app.state.forecasts = ForecastService()


def test_health_and_demo_patients_are_available():
    health_response = health()
    patients_response = patients()
    assert health_response["status"] == "ok"
    assert len(patients_response.data) == 3
    assert patients_response.is_synthetic is True


def test_forecast_and_simulation_include_required_metadata():
    forecast_response = forecast("sunita-d")
    simulated_response = simulate("sunita-d", SimulationRequest(fluid_volume_ml=500, nsaid_withheld=True))
    assert forecast_response.data.forecast.model_version != "unavailable"
    body = simulated_response.data
    assert body.simulated.forecast.risk_48h <= body.baseline.forecast.risk_48h
