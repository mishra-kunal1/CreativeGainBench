# E7 causal contribution controls

**Passed:** not scored (no CUE rows)

**Intended receiver (this run):** `glm-5.1` (PROTOCOL official judge is `llama3.1:8b`; documented cloud override)

**Items on panel:** 24 (synthetic Phase A, seed 42)

**CUE rows:** 0

Scoring stopped after a smoke chat to `https://ollama.com/v1`:

- `error_class=openai.AuthenticationError`
- `http_status=401`
- `error.message=Unauthorized`
- `error.type=api_error`
- `error.code=null`

`OLLAMA_API_KEY` was unset. `CUEBeliefReceiver` fell back to `"ollama"`; that key is not accepted by Ollama Cloud. No CUE values were invented.

PairedMeanDiff rows for {cross, random, irrelevant} are therefore **not available**. Resume-safe path `e7_cue_panel.jsonl` was not created.

Isolation: did not read/write `experiments/experiment1/results/cue_*.jsonl`.
