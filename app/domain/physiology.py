from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from contracts.interfaces import TwinState, TwinValue


@dataclass(frozen=True)
class CalibratedParameters:
    """Dimensionless research parameters for the lumped renal model.

    Values are intentionally model-estimated. They are not clinical measurements.
    """

    ra: float
    re: float
    kf: float
    pbs_mmhg: float
    oncotic_mmhg: float
    renal_venous_pressure_mmhg: float
    reserve: float


def calibrate(egfr: float, uacr_category: str, age_years: float, diabetes: bool, hypertension: bool) -> CalibratedParameters:
    albumin_penalty = {"A1": 0.0, "A2": 0.10, "A3": 0.22}[uacr_category]
    comorbidity_penalty = 0.08 * diabetes + 0.06 * hypertension
    reserve = max(0.08, min(1.0, (egfr / 90.0) - albumin_penalty - comorbidity_penalty - max(0.0, age_years - 45) / 500.0))
    return CalibratedParameters(
        ra=1.00 + (1.0 - reserve) * 0.55,
        re=0.82 + (1.0 - reserve) * 0.25,
        kf=12.5 * (0.35 + 0.65 * reserve),
        pbs_mmhg=12.0,
        oncotic_mmhg=25.0 + (1.0 - reserve) * 2.0,
        renal_venous_pressure_mmhg=5.0,
        reserve=reserve,
    )


def estimate_state(timestamp: datetime, map_mmhg: float, params: CalibratedParameters, ra_shift: float = 0.0, re_shift: float = 0.0) -> TwinState:
    """Evaluate a bounded lumped-parameter renal hemodynamic state.

    The equations preserve useful prototype monotonicity: increasing MAP increases
    Pgc; increasing afferent resistance lowers Pgc. Values are illustrative only.
    """
    ra = max(0.35, params.ra + ra_shift)
    re = max(0.30, params.re + re_shift)
    perfusion = max(0.0, map_mmhg - params.renal_venous_pressure_mmhg)
    rbf = perfusion / (ra + re)
    pgc = max(0.0, map_mmhg - (rbf * ra))
    puf = max(0.0, pgc - params.pbs_mmhg - params.oncotic_mmhg)
    reserve = max(0.0, min(1.0, params.reserve * min(1.0, map_mmhg / 75.0)))

    def estimated(value: float, unit: str, relative_sd: float = 0.12) -> TwinValue:
        return TwinValue(value=round(value, 3), unit=unit, sd=round(max(0.01, abs(value) * relative_sd), 3), provenance="model-estimated")

    return TwinState(
        timestamp=timestamp,
        ra=estimated(ra, "arbitrary resistance units"),
        re=estimated(re, "arbitrary resistance units"),
        rbf=estimated(rbf, "relative flow units", 0.15),
        pgc=estimated(pgc, "mmHg"),
        puf=estimated(puf, "mmHg", 0.18),
        reserve_index=estimated(reserve, "0-1 index", 0.18),
    )
