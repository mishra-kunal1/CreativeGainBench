import Mathlib.Tactic

namespace Creativity.Benchmark

inductive Topology
  | S0_single_agent
  | S1_central_dense
  | S2_decentralized
  | S3_central_sparse
  deriving DecidableEq, Repr

structure EpisodeResult where
  topology : Topology
  milestoneRate : ℝ
  tokenCost : ℕ
  estEntropyB : ℝ
  estNoveltyC : ℝ
  estInteractGain : ℝ
  estimatorEps : ℝ

def predictedS2DominatesS1 (r2 r1 : EpisodeResult) : Prop :=
  r1.estInteractGain + 2 * r1.estimatorEps < r2.estInteractGain

def predictedGroupthinkSignature (r : EpisodeResult)
    (_h : r.topology = .S1_central_dense) : Prop :=
  r.estInteractGain < 0 ∧ r.milestoneRate < 0.7

theorem benchmark_can_detect_S2_dominance
    (trueGainS2 trueGainS1 ε : ℝ)
    (hgap : trueGainS1 + 4 * ε < trueGainS2)
    (_hε : 0 < ε) :
    ∃ (r2 r1 : EpisodeResult),
      r2.topology = .S2_decentralized ∧
      r1.topology = .S1_central_dense ∧
      predictedS2DominatesS1 r2 r1 := by
  let r2 : EpisodeResult :=
    { topology := .S2_decentralized
      milestoneRate := 1
      tokenCost := 0
      estEntropyB := 0
      estNoveltyC := 0
      estInteractGain := trueGainS2 - ε
      estimatorEps := ε }
  let r1 : EpisodeResult :=
    { topology := .S1_central_dense
      milestoneRate := 1
      tokenCost := 0
      estEntropyB := 0
      estNoveltyC := 0
      estInteractGain := trueGainS1 + ε
      estimatorEps := ε }
  refine ⟨r2, r1, rfl, rfl, ?_⟩
  unfold predictedS2DominatesS1
  dsimp [r1, r2]
  linarith

end Creativity.Benchmark
