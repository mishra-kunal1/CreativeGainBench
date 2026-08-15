# Edge-CUE and related engineering extensions

This note documents what the Python MAS scoring path measures that **is not**
covered by theorems in `math_backing/`.

## Covered by Lean (do not over-claim)

| Object | Lean locus | Scope |
|--------|------------|--------|
| `CUEModel`, Brier Δ / bits, `cue_gate` | `Creativity/CUE/*` | Single output / belief update |
| Step-CUE curve / DC | `Creativity/CUE/Trajectory.lean` | Same-agent trajectory snapshots |
| \(G_k\) → joint CUE dominance | `Creativity/CUE/MASBridge.lean` (`Gk_implies_CUE_improvement`, PROOF-07) | Under `ReceiverCalibratedMAS` and gain \(> 2\varepsilon_{\mathrm{cal}}\) |
| ProbeCompressor \(R_D\) | `Creativity/D/*` | Count-based deformation (Python: `count_ngram` / `deformation`) |

The Python package uses **only** CountNgram ProbeCompressor deformation for
\(R_D\). The former KenLM prefix-conditioning proxy has been removed.

## Not covered by any theorem (engineering extensions)

| Object | Python locus | Notes |
|--------|--------------|--------|
| **Edge-CUE** | `metrics/edge_cue.py` | Instantiates existing `CUEModel` on a handoff with a *conditioned prior* (belief after upstream text). Formula unchanged; elicitation granularity is new. |
| **HandoffGain rate** | mean of `cue_gate` over registered edges | Aggregate KPI; not a Lean definition. |
| **Critic ablation ΔCUE** | `eval/mas_ablation.py` | Causal check beyond observational belief movement; explicitly out of formal scope. |
| Empirical PROOF-07 check | `eval/mas_bridge_check.py` | Reports whether measured \(G_k\) rows satisfy `joint_cue > max(solo_cues)`; does not prove calibration. |

## Deferred formalization

A natural Lean target (`PROOF-07b`-style) would relate positive edge-CUE along a
heterogeneous multi-agent chain to a lower bound on terminal CUE. Do **not**
formalize until Phase 3/4 empirics show the edge definition behaves sensibly.
