from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.interfaces import StaticProfile, TelemetryFrame


NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def _frames(map_day: float, map_night: float, hr: float, hrv: float, rising_hr: bool = False) -> list[TelemetryFrame]:
    frames: list[TelemetryFrame] = []
    for elapsed in range(48):
        timestamp = NOW - timedelta(hours=47 - elapsed)
        night = timestamp.hour >= 22 or timestamp.hour < 6
        adjusted_hr = hr + (elapsed / 8 if rising_hr else 0)
        map_value = map_night if night else map_day
        frames.append(TelemetryFrame(
            timestamp=timestamp,
            map_mmhg=map_value,
            sbp_mmhg=map_value + 35,
            dbp_mmhg=map_value - 15,
            hr_bpm=round(adjusted_hr, 1),
            hrv_rmssd_ms=hrv,
            spo2_pct=97.0,
            resp_rate_bpm=16.0,
            sleep_stage="deep" if night else "awake",
        ))
    return frames


DEMO_PATIENTS = {
    "ramesh-k": {
        "display_name": "Ramesh K.", "ckd_stage": "G3a", "trend": "flat",
        "profile": StaticProfile(patient_id="ramesh-k", age_years=58, sex="M", baseline_creatinine_mg_dl=1.5, egfr_ml_min_1_73m2=52, uacr_category="A2", diabetes=True, hypertension=True, on_acei_arb=True),
        "telemetry": _frames(82, 72, 72, 32),
    },
    "sunita-d": {
        "display_name": "Sunita D.", "ckd_stage": "G3b", "trend": "up",
        "profile": StaticProfile(patient_id="sunita-d", age_years=63, sex="F", baseline_creatinine_mg_dl=2.1, egfr_ml_min_1_73m2=28, uacr_category="A3", diabetes=True, hypertension=True, on_acei_arb=True, nsaid_exposure=True),
        "telemetry": _frames(63, 61, 86, 12, rising_hr=True),
    },
    "mohan-l": {
        "display_name": "Mohan L.", "ckd_stage": "G3b", "trend": "down",
        "profile": StaticProfile(patient_id="mohan-l", age_years=67, sex="M", baseline_creatinine_mg_dl=1.9, egfr_ml_min_1_73m2=34, uacr_category="A2", diabetes=True, hypertension=True, on_diuretic=True),
        "telemetry": _frames(76, 68, 76, 26),
    },
}
