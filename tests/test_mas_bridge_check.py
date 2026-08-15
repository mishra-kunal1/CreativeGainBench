from creativegainbench.eval.mas_bridge_check import check_rows


def test_bridge_holds_when_joint_dominates():
    rows = [
        {"g_k": 1.0, "joint_cue": 0.5, "solo_cues": [0.1, 0.2]},
        {"g_k": 1.0, "joint_cue": 0.05, "solo_cues": [0.1, 0.2]},
    ]
    s = check_rows(rows, gk_threshold=0.0)
    assert s["n_eligible"] == 2
    assert s["n_hold"] == 1
    assert s["pass_rate"] == 0.5


def test_no_eligible():
    s = check_rows([{"g_k": 0.0, "joint_cue": 1.0, "solo_cues": [0.1]}], gk_threshold=0.5)
    assert s["n_eligible"] == 0
    assert s["pass_rate"] is None
