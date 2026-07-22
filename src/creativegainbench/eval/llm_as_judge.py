import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_PATH = Path("data/results/infinity_chat_results.jsonl")
OUTPUT_PATH = Path("data/evaluation/creativity_judge_scores.jsonl")

# Capable, free-tier, non-GPT models on OpenRouter.
# Using the smaller Nemotron variant (120B vs 550B) since it's noticeably faster per call.
JUDGE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "cohere/north-mini-code:free",
]

# Free-tier models get rate-limited (429) often; retry a couple times with backoff
# instead of dropping the judge's score for that response.
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

CRITERIA = ["novelty", "surprise", "usefulness", "coherence"]

JUDGE_PROMPT = """You are an expert judge evaluating the creativity of an AI response to a prompt.

Prompt:
{prompt}

Response:
{response}

Rate the response on each of these criteria, using an integer score from 1 (very poor) to 10 (excellent):
- novelty: how original and non-generic the idea/content is
- surprise: how unexpected the response is compared to a typical/predictable answer
- usefulness: how practically valuable or relevant the response is to the prompt
- coherence: how well-formed, clear, and internally consistent the response is

Respond with ONLY a JSON object, no other text, in this exact format:
{{"novelty": <int>, "surprise": <int>, "usefulness": <int>, "coherence": <int>}}
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def judge_response(model: str, prompt: str, response: str) -> dict:
    # Retry only on 429 (rate limit) - other errors (bad parse, etc.) shouldn't be retried.
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, response=response)}
                ],
                temperature=0,
            )
            break
        except Exception as e:
            is_last_attempt = attempt == MAX_RETRIES - 1
            if "429" in str(e) and not is_last_attempt:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise

    text = completion.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse judge output from {model}: {text}")
    scores = json.loads(match.group(0))
    return {criterion: float(scores[criterion]) for criterion in CRITERIA}


def main():
    records = [json.loads(line) for line in INPUT_PATH.read_text().splitlines() if line.strip()]
    total_responses = sum(len(r["responses"]) for r in records)

    results = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write each response's result as soon as it's judged, so progress survives a crash
    # instead of being lost if something fails near the end of a long run.
    with OUTPUT_PATH.open("w") as out_file:
        done = 0
        for record in records:
            prompt = record["prompt"]
            for response_dict in record["responses"]:
                response_key, response_text = next(iter(response_dict.items()))

                judge_scores = {}
                for model in JUDGE_MODELS:
                    try:
                        judge_scores[model] = judge_response(model, prompt, response_text)
                    except Exception as e:
                        print(f"[warn] {model} failed on {response_key!r}: {e}")

                done += 1
                if not judge_scores:
                    print(f"[{done}/{total_responses}] {response_key}: all judges failed, skipping")
                    continue

                mean_scores = {
                    criterion: statistics.mean(s[criterion] for s in judge_scores.values())
                    for criterion in CRITERIA
                }
                mean_scores["overall"] = statistics.mean(mean_scores.values())

                result = {
                    "prompt": prompt,
                    "response_key": response_key,
                    "judge_scores": judge_scores,
                    "mean_scores": mean_scores,
                }
                results.append(result)
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
                print(f"[{done}/{total_responses}] {response_key}: overall={mean_scores['overall']:.2f}")

    print(f"Judged {len(results)} responses across {len(records)} prompts.")
    print(f"Overall mean creativity score: {statistics.mean(r['mean_scores']['overall'] for r in results):.2f}")
    for criterion in CRITERIA:
        crit_mean = statistics.mean(r["mean_scores"][criterion] for r in results)
        print(f"  {criterion}: {crit_mean:.2f}")


if __name__ == "__main__":
    main()
