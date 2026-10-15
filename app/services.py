from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import joblib

from app.demo import DEMO_PATIENTS
from app.domain.physiology import calibrate, estimate_state
from app.schemas import AssumptionEntry, ForecastResponse, SimulationRequest, TrajectoryPoint
from contracts.interfaces import ForecastResult, TrajectoryPerturbation
from ml.features.api import extract_features


ASSUMPTIONS_VERSION = "v0.1"
ASSUMPTIONS = [
    AssumptionEntry(key="iv_fluid", central="+6 mmHg MAP per 500 mL, capped", low_high="+3 to +9 mmHg", rationale="Prototype saturating volume-response mapping over the requested duration.", evidence_grade="prototype"),
    AssumptionEntry(key="nsaid_withheld", central="Ra −0.15 over 12 h", low_high="−0.08 to −0.22", rationale="Research-only directional afferent resistance assumption.", evidence_grade="prototype"),
    AssumptionEntry(key="acei_arb_paused", central="Re +0.10 over 12 h", low_high="+0.05 to +0.15", rationale="Research-only directional efferent resistance assumption.", evidence_grade="prototype"),
    AssumptionEntry(key="renal_vasodilator", central="Ra −0.08 over 6 h", low_high="−0.02 to −0.15", rationale="Exploratory generic intervention with wide uncertainty.", evidence_grade="prototype"),
]


class ForecastService:
    def __init__(self) -> None:
        self.forecaster = None
        self.load_error: str | None = None
        try:
            artifact = Path(__file__).resolve().parents[1] / "ml/artefacts/v1/model.joblib"
            self.forecaster = joblib.load(artifact)
        except Exception as exc:  
            self.load_error = str(exc)

    @property
    def model_version(self) -> str:
        return self.forecaster.meta.model_version if self.forecaster else "unavailable"

    def _require_model(self):
        if not self.forecaster:
            raise RuntimeError(f"Forecaster artifact unavailable: {self.load_error}")
        return self.forecaster

    def features_and_state(self, patient_id: str):
        patient = DEMO_PATIENTS[patient_id]
        now = patient["telemetry"][-1].timestamp
        features = extract_features(patient["profile"], patient["telemetry"], now)
        params = calibrate(patient["profile"].egfr_ml_min_1_73m2, patient["profile"].uacr_category, patient["profile"].age_years, patient["profile"].diabetes, patient["profile"].hypertension)
        state = estimate_state(now, patient["telemetry"][-1].map_mmhg or 75.0, params)
        return patient, now, features, params, state

    def forecast(self, patient_id: str, perturbation: TrajectoryPerturbation | None = None, request: SimulationRequest | None = None) -> tuple[ForecastResponse, object]:
        patient, now, features, params, state = self.features_and_state(patient_id)
        forecaster = self._require_model()
        result: ForecastResult = forecaster.predict_trajectory(features, perturbation, patient_id, now) if perturbation else forecaster.predict(features, patient_id, now)
        map_delta, ra_shift, re_shift = _effect_at_horizon(request)
        if request:
            state = estimate_state(
                now,
                (patient["telemetry"][-1].map_mmhg or 75.0) + map_delta,
                params,
                ra_shift,
                re_shift,
            )
        points = []
        for hour in range(0, 49, 6):
            fraction = min(1.0, hour / max(1.0, request.fluid_duration_hours if request else 1.0)) if request else 0.0
            point_state = estimate_state(now + timedelta(hours=hour), (patient["telemetry"][-1].map_mmhg or 75) + map_delta * fraction, params, ra_shift * min(1.0, hour / 12), re_shift * min(1.0, hour / 12))
            egfr_proxy = max(1.0, patient["profile"].egfr_ml_min_1_73m2 * (0.55 + 0.45 * point_state.reserve_index.value))
            points.append(TrajectoryPoint(timestamp=point_state.timestamp, pgc_mmhg=point_state.pgc.value, pgc_interval=(max(0, point_state.pgc.value - point_state.pgc.sd), point_state.pgc.value + point_state.pgc.sd), egfr_proxy=round(egfr_proxy, 2), egfr_proxy_interval=(round(max(0, egfr_proxy * .82), 2), round(egfr_proxy * 1.18, 2))))
        return ForecastResponse(forecast=result, projected_trajectory=points), state


def _effect_at_horizon(request: SimulationRequest | None) -> tuple[float, float, float]:
    if request is None:
        return 0.0, 0.0, 0.0
    scenario = {"low": 0.65, "central": 1.0, "high": 1.35}[request.scenario]
    fluid_effect = min(9.0, request.fluid_volume_ml / 500 * 6.0) * scenario
    map_delta = request.map_target_delta_mmhg + fluid_effect + (2.0 if request.diuretic_paused else 0.0)
    ra_shift = (-0.15 if request.nsaid_withheld else 0.0) + (-0.08 if request.renal_vasodilator_started else 0.0)
    re_shift = 0.10 if request.acei_arb_paused else 0.0
    return map_delta, ra_shift * scenario, re_shift * scenario


def make_perturbation(request: SimulationRequest) -> TrajectoryPerturbation:
    map_delta, _, _ = _effect_at_horizon(request)
    overrides = {"nsaid_exposure": 0.0} if request.nsaid_withheld else {}
    deltas = {"map_mean_24h": map_delta, "map_min_24h": map_delta, "hypotension_burden_min": -min(240.0, map_delta * 20)}
    return TrajectoryPerturbation(feature_overrides=overrides, feature_deltas=deltas, scenario=request.scenario)


def simulation_warnings(patient_id: str, request: SimulationRequest) -> list[str]:
    patient = DEMO_PATIENTS[patient_id]["profile"]
    warnings = ["Exploratory simulation only — it is not a treatment recommendation."]
    if request.fluid_volume_ml >= 1000 or (patient.on_diuretic and request.fluid_volume_ml > 500):
        warnings.append("Volume-overload plausibility warning: review cardiac and volume status outside this model.")
    if request.renal_vasodilator_started:
        warnings.append("Renal vasodilator effects have wide prototype uncertainty.")
    return warnings


def audit_id() -> str:
    return f"sim-{uuid4().hex[:12]}"
