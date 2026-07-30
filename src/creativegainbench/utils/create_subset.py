"""
Create evaluation subsets.

Delegates to build_eval_prompts (HF data if present, else packaged held-out bank)
with frozen-probe decontamination.
"""

from creativegainbench.utils.build_eval_prompts import main


if __name__ == "__main__":
    main()
