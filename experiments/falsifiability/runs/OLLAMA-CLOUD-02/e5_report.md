# E5 receiver + encoder stability

**Passed (E5a ∧ E5b):** False

## E5a receivers
Passed: False  ·  n=24  ·  non-inert=['gemma4:31b', 'gpt-oss:20b', 'nemotron-3-nano:30b']
cue_inert (excluded from Spearman/CCC, not a ρ-bar fail): []

- gemma4:31b vs gpt-oss:20b: Spearman ρ=0.880 CI=[0.654, 0.977] indeterminate; CCC=0.904 CI=[0.678, 0.965] indeterminate; pass=False
- gemma4:31b vs nemotron-3-nano:30b: Spearman ρ=0.767 CI=[0.382, 0.902] indeterminate; CCC=0.916 CI=[0.729, 0.971] indeterminate; pass=False
- gpt-oss:20b vs nemotron-3-nano:30b: Spearman ρ=0.789 CI=[0.581, 0.916] indeterminate; CCC=0.865 CI=[0.622, 0.945] indeterminate; pass=False
- Krippendorff α (optional): 0.895 CI=[0.748, 0.953] indeterminate

## E5b encoder
Passed: False
- mean CV original/plain/technical = 3.9301649934895857 CI=[2.5081308816285897, 6.4236701949278645] pass=False
- scrambled negative control mean CV = 0.8925508392602929 CI=[0.6819848363309907, 1.210967540990653] (must NOT pass; control_ok=True)

