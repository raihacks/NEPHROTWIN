from datetime import datetime, timezone

from app.domain.physiology import calibrate, estimate_state


def test_higher_map_increases_pgc_under_fixed_resistances():
    params = calibrate(45, "A2", 60, True, True)
    low = estimate_state(datetime.now(timezone.utc), 65, params)
    high = estimate_state(datetime.now(timezone.utc), 80, params)
    assert high.pgc.value > low.pgc.value


def test_higher_afferent_resistance_lowers_pgc():
    params = calibrate(45, "A2", 60, True, True)
    baseline = estimate_state(datetime.now(timezone.utc), 75, params)
    constricted = estimate_state(datetime.now(timezone.utc), 75, params, ra_shift=0.25)
    assert constricted.pgc.value < baseline.pgc.value
    assert constricted.pgc.provenance == "model-estimated"
