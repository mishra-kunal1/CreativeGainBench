# E5 receiver + encoder stability

**Passed (E5a ∧ E5b):** False

## E5a receivers
Passed: False  ·  n=24  ·  ['gemma4:31b', 'gpt-oss:20b', 'nemotron-3-nano:30b']

- gemma4:31b vs gpt-oss:20b: Spearman ρ=nan CI=[nan, nan] indeterminate; CCC=0.000 CI=[0.000, 0.000] different; pass=False
- gemma4:31b vs nemotron-3-nano:30b: Spearman ρ=0.101 CI=[-0.293, 0.483] different; CCC=-0.009 CI=[-0.080, 0.003] different; pass=False
- gpt-oss:20b vs nemotron-3-nano:30b: Spearman ρ=nan CI=[nan, nan] indeterminate; CCC=0.000 CI=[0.000, 0.000] different; pass=False
- Krippendorff α (optional): -0.285 CI=[-0.336, -0.218] different

## E5b encoder
Passed: False
- mean CV original/plain/technical = 1.7950801701613555 CI=[1.7950801701613555, 1.7950801701613555] pass=False
- scrambled negative control mean CV = 1.3417915843793475 CI=[1.3417915843793475, 1.3417915843793475] (must NOT pass; control_ok=True)

