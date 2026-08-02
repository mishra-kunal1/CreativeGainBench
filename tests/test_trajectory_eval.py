"""Trajectory extraction and per-row scoring tests. No network, no artifact loading."""

from __future__ import annotations

from creativegainbench.eval.trajectory_eval import (
    _extract_trajectory_texts,
    score_trajectory_for_row,
)


def _transcript(n_rounds: int) -> list[dict]:
    t = [
        {"role": "proposer", "step": "solo", "round": None, "model": "m", "content": "solo-p"},
        {"role": "critic", "step": "solo", "round": None, "model": "m", "content": "solo-c"},
        {"role": "verifier", "step": "solo", "round": None, "model": "m", "content": "solo-v"},
        {"role": "proposer", "step": "draft", "round": None, "model": "m", "content": "draft-text"},
    ]
    for r in range(1, n_rounds + 1):
        t.append({"role": "critic", "step": "critique", "round": r, "model": "m", "content": f"critique-{r}"})
        t.append({"role": "proposer", "step": "revision", "round": r, "model": "m", "content": f"revision-{r}"})
        t.append({"role": "verifier", "step": "verify", "round": r, "model": "m", "content": f"verify-{r}"})
    return t


def test_extract_trajectory_texts_draft_plus_revisions_in_order():
    texts = _extract_trajectory_texts(_transcript(n_rounds=3))
    assert texts == ["draft-text", "revision-1", "revision-2", "revision-3"]


def test_extract_trajectory_texts_draft_only_when_no_revisions():
    texts = _extract_trajectory_texts(_transcript(n_rounds=0))
    assert texts == ["draft-text"]


def test_extract_trajectory_texts_ignores_solo_critique_verify_steps():
    texts = _extract_trajectory_texts(_transcript(n_rounds=1))
    assert "solo-p" not in texts
    assert "critique-1" not in texts
    assert "verify-1" not in texts


def test_extract_trajectory_texts_handles_out_of_order_transcript():
    # Revisions recorded out of round order should still be sorted correctly.
    t = [
        {"role": "proposer", "step": "draft", "round": None, "content": "d"},
        {"role": "proposer", "step": "revision", "round": 2, "content": "r2"},
        {"role": "proposer", "step": "revision", "round": 1, "content": "r1"},
    ]
    assert _extract_trajectory_texts(t) == ["d", "r1", "r2"]


def test_score_trajectory_for_row_full_pipeline():
    row = {
        "prompt": "write something",
        "domain": "creative_writing",
        "transcript": _transcript(n_rounds=4),
    }

    calls = {"cue": [], "rb": []}

    def cue_fn(prompt: str, text: str) -> float:
        calls["cue"].append((prompt, text))
        # Monotone-ish rising CUE with saturation, deterministic given text.
        step = int(text.rsplit("-", 1)[-1]) if "-" in text and text.split("-")[-1].isdigit() else 0
        return 0.1 * (step + 1)

    def rb_fn(text: str) -> float:
        calls["rb"].append(text)
        return 0.5

    result = score_trajectory_for_row(row, cue_fn=cue_fn, rb_fn=rb_fn)

    assert result["prompt"] == "write something"
    assert result["domain"] == "creative_writing"
    assert result["n_steps"] == 5  # draft + 4 revisions
    assert result["t_values"] == [1, 2, 3, 4, 5]
    assert len(result["step_cue"]) == 5
    assert len(result["step_rb"]) == 5
    assert result["step_cue_fit"]["n_points"] == 5
    assert result["diverge_converge"]["n_points"] == 5
    # cue_fn/rb_fn called once per trajectory step, in step order.
    assert [text for _, text in calls["cue"]] == ["draft-text", "revision-1", "revision-2", "revision-3", "revision-4"]
    assert calls["rb"] == ["draft-text", "revision-1", "revision-2", "revision-3", "revision-4"]


def test_score_trajectory_for_row_missing_transcript_returns_none():
    row = {"prompt": "p", "domain": "creative_writing"}
    result = score_trajectory_for_row(row, cue_fn=lambda p, y: 0.0, rb_fn=lambda y: 0.0)
    assert result is None


def test_score_trajectory_for_row_missing_prompt_returns_none():
    row = {"transcript": _transcript(n_rounds=1)}
    result = score_trajectory_for_row(row, cue_fn=lambda p, y: 0.0, rb_fn=lambda y: 0.0)
    assert result is None


def test_score_trajectory_for_row_two_points_flags_unreliable_fit():
    row = {
        "prompt": "p",
        "domain": "creative_writing",
        "transcript": _transcript(n_rounds=1),
    }
    result = score_trajectory_for_row(row, cue_fn=lambda p, y: 0.1, rb_fn=lambda y: 0.5)
    assert result["n_steps"] == 2
    assert result["step_cue_fit"]["reliable"] is False
    assert result["diverge_converge"]["dc"] == 0  # < 3 points, definitionally 0
