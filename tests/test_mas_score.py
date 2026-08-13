from creativegainbench.eval.mas_score import score_mas_row
from creativegainbench.metrics.edge_cue import handoff_gain_rate


def test_score_mas_row_uses_existing_chain():
    row = {
        "prompt": "task",
        "agent_texts": ["a", "b", "c"],
        "joint_text": "joint",
        "transcript": [
            {"role": "proposer", "step": "draft", "content": "d1"},
            {"role": "proposer", "step": "revision", "round": 1, "content": "r1"},
        ],
        "edge_cue_chain": [
            {
                "edge_id": "proposer_draft_to_revision",
                "cue": 0.1,
                "gate": 1.0,
                "diagnostic": False,
            },
            {
                "edge_id": "proposer_revision_to_verifier",
                "cue": 0.0,
                "gate": 0.0,
                "diagnostic": True,
            },
        ],
    }

    def cue_fn(p, y):
        return 0.01

    def rb_fn(y):
        return 0.5

    out = score_mas_row(row, cue_fn=cue_fn, rb_fn=rb_fn)
    assert out["handoff_gain_rate"] == handoff_gain_rate(
        [e for e in row["edge_cue_chain"] if not e["diagnostic"]]
    )
    assert out["step_cue"]["n_steps"] == 2
