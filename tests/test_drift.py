"""Tests for monitoring/drift.py."""

from monitoring.drift import DriftDetector


SAMPLE_STATS = {
    "return_1d": {"mean": 0.001, "std": 0.015},
    "volatility_5d": {"mean": 0.012, "std": 0.005},
    "rsi_14": {"mean": 55.0, "std": 10.0},
}


def test_no_drift_within_bounds():
    """Values within 3σ should not trigger drift."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({"return_1d": 0.02, "volatility_5d": 0.010})
    assert not result["drift_detected"]
    assert result["drifted_features"] == []


def test_drift_detected_beyond_threshold():
    """A value >3σ from the mean should be flagged."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    # return_1d mean=0.001, std=0.015 → 3σ boundary = 0.001 ± 0.045
    # 0.10 is well beyond 3σ
    result = detector.check({"return_1d": 0.10})
    assert result["drift_detected"]
    assert "return_1d" in result["drifted_features"]


def test_multiple_features_flagged():
    """Multiple out-of-bounds features should all appear in drifted_features."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({
        "return_1d": 0.10,       # drifted: z = (0.10 - 0.001) / 0.015 ≈ 6.6
        "rsi_14": 110.0,         # drifted: z = (110 - 55) / 10 = 5.5
        "volatility_5d": 0.012,  # normal: z = 0
    })
    assert result["drift_detected"]
    assert set(result["drifted_features"]) == {"return_1d", "rsi_14"}


def test_unknown_feature_skipped():
    """Features not in training stats are silently skipped, not treated as drift."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({"new_feature_not_in_stats": 999.9})
    assert not result["drift_detected"]
    assert result["drifted_features"] == []
