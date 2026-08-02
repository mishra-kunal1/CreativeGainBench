# CreativeGainBench cost estimate

## Run metadata

- **Generated at (UTC):** 2026-07-26T18:10:02.239224+00:00
- **Sample size per domain:** 10
- **Completions per prompt (`n`):** 5
- **Domains:** infinity_chat, formalmath, rinobench
- **Assumed generation completion tokens:** 800
- **Assumed judge completion tokens:** 80
- **Token heuristic:** `ceil(len(text)/4)`
- **OpenRouter prices fetched at:** 2026-07-26T18:10:02.088142+00:00
- **Ollama Cloud tags fetched at:** 2026-07-26T18:10:02.223291+00:00

## Assumptions

- Token counts use ceil(len(text)/4); not provider-native tokenizers.
- Generation assumes one chat request per prompt with n completions (input billed once).
- Assumed generation completion tokens per completion: 800.
- Assumed judge completion tokens per judge call: 80.
- Judge cost = sample prompts × n × number of judge models × (judge input + output prices).
- Open-model gen USD uses OpenRouter sibling pricing as a research cross-quote.
- Ollama Cloud itself is subscription + GPU-time quota — see plan table below.
- No live generation/judge inference is performed for this estimate.

## Per-domain estimates (sample)

### infinity_chat

- Prompts in sample: **10** (mean input tokens ≈ 30.7)
- Full subset size (for scale-up): **300**

| Model | Provider | Available | Gen USD | Judge USD | Total USD | Gen calls | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-4o mini | openrouter | yes | $0.0240 | $0.0000 | $0.0240 | 50 |  |
| GPT-4.1 | openrouter | yes | $0.3206 | $0.0000 | $0.3206 | 50 |  |
| Claude Sonnet 5 | openrouter | yes | $0.4006 | $0.0000 | $0.4006 | 50 |  |
| Claude Opus 5 | openrouter | yes | $1.0015 | $0.0000 | $1.0015 | 50 |  |
| Claude Fable 5 | openrouter | yes | $2.0031 | $0.0000 | $2.0031 | 50 |  |
| Gemini 2.5 Flash | openrouter | yes | $0.1001 | $0.0000 | $0.1001 | 50 |  |
| Gemini 2.5 Pro | openrouter | yes | $0.4004 | $0.0000 | $0.4004 | 50 |  |
| Llama 4 Maverick | ollama_cloud | no | $0.0321 | $0.0000 | $0.0321 | 50 | not listed on Ollama Cloud tags as `llama4`; Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `meta-llama/llama-4-maverick` |
| DeepSeek V4 Flash | ollama_cloud | yes | $0.0112 | $0.0000 | $0.0112 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `deepseek/deepseek-v4-flash` |
| Kimi K2.6 | ollama_cloud | yes | $0.1090 | $0.0000 | $0.1090 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `moonshotai/kimi-k2.6` |
| GLM 5.1 | ollama_cloud | yes | $0.1217 | $0.0000 | $0.1217 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `z-ai/glm-5.1` |
| Qwen 3.5 397B | ollama_cloud | yes | $0.0937 | $0.0000 | $0.0937 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `qwen/qwen3.5-397b-a17b` |

### formalmath

- Prompts in sample: **10** (mean input tokens ≈ 75.6)
- Full subset size (for scale-up): **300**

| Model | Provider | Available | Gen USD | Judge USD | Total USD | Gen calls | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-4o mini | openrouter | yes | $0.0241 | $0.0000 | $0.0241 | 50 |  |
| GPT-4.1 | openrouter | yes | $0.3215 | $0.0000 | $0.3215 | 50 |  |
| Claude Sonnet 5 | openrouter | yes | $0.4015 | $0.0000 | $0.4015 | 50 |  |
| Claude Opus 5 | openrouter | yes | $1.0038 | $0.0000 | $1.0038 | 50 |  |
| Claude Fable 5 | openrouter | yes | $2.0076 | $0.0000 | $2.0076 | 50 |  |
| Gemini 2.5 Flash | openrouter | yes | $0.1002 | $0.0000 | $0.1002 | 50 |  |
| Gemini 2.5 Pro | openrouter | yes | $0.4009 | $0.0000 | $0.4009 | 50 |  |
| Llama 4 Maverick | ollama_cloud | no | $0.0322 | $0.0000 | $0.0322 | 50 | not listed on Ollama Cloud tags as `llama4`; Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `meta-llama/llama-4-maverick` |
| DeepSeek V4 Flash | ollama_cloud | yes | $0.0113 | $0.0000 | $0.0113 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `deepseek/deepseek-v4-flash` |
| Kimi K2.6 | ollama_cloud | yes | $0.1093 | $0.0000 | $0.1093 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `moonshotai/kimi-k2.6` |
| GLM 5.1 | ollama_cloud | yes | $0.1222 | $0.0000 | $0.1222 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `z-ai/glm-5.1` |
| Qwen 3.5 397B | ollama_cloud | yes | $0.0939 | $0.0000 | $0.0939 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `qwen/qwen3.5-397b-a17b` |

### rinobench

- Prompts in sample: **10** (mean input tokens ≈ 286.6)
- Full subset size (for scale-up): **299**

| Model | Provider | Available | Gen USD | Judge USD | Total USD | Gen calls | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-4o mini | openrouter | yes | $0.0244 | $0.0000 | $0.0244 | 50 |  |
| GPT-4.1 | openrouter | yes | $0.3257 | $0.0000 | $0.3257 | 50 |  |
| Claude Sonnet 5 | openrouter | yes | $0.4057 | $0.0000 | $0.4057 | 50 |  |
| Claude Opus 5 | openrouter | yes | $1.0143 | $0.0000 | $1.0143 | 50 |  |
| Claude Fable 5 | openrouter | yes | $2.0287 | $0.0000 | $2.0287 | 50 |  |
| Gemini 2.5 Flash | openrouter | yes | $0.1009 | $0.0000 | $0.1009 | 50 |  |
| Gemini 2.5 Pro | openrouter | yes | $0.4036 | $0.0000 | $0.4036 | 50 |  |
| Llama 4 Maverick | ollama_cloud | no | $0.0326 | $0.0000 | $0.0326 | 50 | not listed on Ollama Cloud tags as `llama4`; Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `meta-llama/llama-4-maverick` |
| DeepSeek V4 Flash | ollama_cloud | yes | $0.0116 | $0.0000 | $0.0116 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `deepseek/deepseek-v4-flash` |
| Kimi K2.6 | ollama_cloud | yes | $0.1107 | $0.0000 | $0.1107 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `moonshotai/kimi-k2.6` |
| GLM 5.1 | ollama_cloud | yes | $0.1242 | $0.0000 | $0.1242 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `z-ai/glm-5.1` |
| Qwen 3.5 397B | ollama_cloud | yes | $0.0947 | $0.0000 | $0.0947 | 50 | Ollama Cloud: subscription/GPU-time quota (not $/token); USD column = OpenRouter cross-quote `qwen/qwen3.5-397b-a17b` |

## Provider / family rollup (sample)

| Family | Provider | Gen USD | Judge USD | Total USD | Gen calls |
| --- | --- | ---: | ---: | ---: | ---: |
| deepseek | ollama_cloud | $0.0342 | $0.0000 | $0.0342 | 150 |
| glm | ollama_cloud | $0.3681 | $0.0000 | $0.3681 | 150 |
| kimi | ollama_cloud | $0.3289 | $0.0000 | $0.3289 | 150 |
| llama | ollama_cloud | $0.0968 | $0.0000 | $0.0968 | 150 |
| qwen | ollama_cloud | $0.2823 | $0.0000 | $0.2823 | 150 |
| claude | openrouter | $10.2668 | $0.0000 | $10.2668 | 450 |
| gemini | openrouter | $1.5061 | $0.0000 | $1.5061 | 300 |
| gpt | openrouter | $1.0404 | $0.0000 | $1.0404 | 300 |

## Grand totals and scale-up

- **Sample gen USD (OpenRouter-priced rows):** $13.9237
- **Sample judge USD:** $0.0000
- **Sample total USD:** $13.9237
- **Projected full-subset gen USD** (linear × full/sample): $417.2419
- **Projected full-subset judge USD:** $0.0000
- **Projected full-subset total USD:** $417.2419

Scale-up is linear in prompt count and assumes the same mean input length and output-token assumptions. Ollama Cloud open-model dollars above use OpenRouter cross-quotes only; actual Ollama Cloud spend is subscription/quota.

## Ollama Cloud plans (open models)

| Plan | Monthly USD | Concurrent models | Notes |
| --- | ---: | ---: | --- |
| Free | $0 | 1 | Light cloud usage; session/weekly quotas apply (GPU-time, not tokens). |
| Pro | $20 | 3 | ~50x Free cloud usage; suitable for day-to-day open-model workloads. |
| Max | $100 | 10 | Highest included usage; heavy concurrent / sustained agent workloads. |

- Ollama Cloud bills via subscription + GPU-time quota (not $/token). Sample generation calls≈50; linear full-subset projection≈1500. Free may cover a tiny smoke sample; Pro is safer for repeated runs. See https://ollama.com/pricing [infinity_chat / Llama 4 Maverick]
- Ollama Cloud bills via subscription + GPU-time quota (not $/token). Sample generation calls≈50; linear full-subset projection≈1495. Free may cover a tiny smoke sample; Pro is safer for repeated runs. See https://ollama.com/pricing [rinobench / Llama 4 Maverick]

## API / plan checklist

- `OPENROUTER_API_KEY` recommended for live $/token (models list often works without it).
- `OLLAMA_API_KEY` for Ollama Cloud open-model calls (tags may be public).
- Judges currently use OpenRouter free models → judge USD often $0.
- Ollama Cloud open-model USD in this report is an OpenRouter cross-quote only.
- Install subsets via `download-datasets` then `create-subset` before estimating.
