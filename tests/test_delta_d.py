from creativegainbench.metrics.delta_d import (
    d_gate,
    quantile,
    resolve_delta_d,
    thresholds_from_negatives,
)


def test_quantile_and_thresholds():
    vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert quantile(vals, 0.5) == 0.2
    thr = thresholds_from_negatives({"0": vals}, q=0.8, eps=0.0)
    assert thr["0"]["n_neg"] == 5
    assert thr["0"]["delta_d_95"] == quantile(vals, 0.8)


def test_resolve_and_gate():
    thr = {"0": {"delta_d_95": 0.1}, "default": {"delta_d_95": 0.2}}
    assert resolve_delta_d(thr, 0) == 0.1
    assert resolve_delta_d(thr, 99) == 0.2
    assert d_gate(0.15, 0.1) == 1.0
    assert d_gate(0.05, 0.1) == 0.0
    assert d_gate(0.15, 0.1, feasible=False) == 0.0


def test_feasibility_bit():
    from creativegainbench.metrics.feasibility import feasibility_bit

    assert feasibility_bit("line one\nline two") is True
    assert feasibility_bit("one line only") is False
    assert feasibility_bit("  \n  ") is False
