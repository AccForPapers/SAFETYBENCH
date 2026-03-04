# SafetyBench

A safety benchmark for evaluating LLM toxicity and jailbreak robustness in Russian, grounded in Russian legislation.

The benchmark covers 8 toxicity categories organized into two macroclasses (illegal content under Russian law and universally toxic content) and includes adversarial prompts from 5 jailbreak strategies.

## Disclaimer

This dataset contains examples that may be disturbing, harmful, or upsetting. It includes a range of sensitive topics such as discriminatory language, discussions of abuse, violence, self-harm, sexual content, and misinformation, among other high-risk categories. The primary purpose of this data is to advance research in building safe LLMs. It is not recommended to train an LLM exclusively on the harmful examples contained in this dataset.

## Files

| File | Description |
|------|-------------|
| `benchmark_original.jsonl` | Base benchmark — 400 direct toxic prompts across 8 taxonomy categories |
| `benchmark_jailbreaks.jsonl` | Jailbreak benchmark — 800 prompts wrapped in adversarial prefixes from 5 jailbreak strategies |
| `run_benchmark.py` | Two-stage evaluation script: generate model responses and judge them for refusal |

## Usage

### Prerequisites

```bash
pip install httpx python-dotenv tqdm
export OPENROUTER_API_KEY="your-key-here"
```

### Stage 1: Generate responses

Send benchmark prompts to a target model and save its responses:

```bash
# Base benchmark
python run_benchmark.py generate --benchmark original --model <model_id>

# Jailbreak benchmark
python run_benchmark.py generate --benchmark jailbreaks --model <model_id>
```

Options:
- `--model` — target model ID on OpenRouter (default: `qwen/qwen3-235b-a22b-2507`)
- `--concurrency` — max parallel requests (default: 15)
- `--output-dir` — output directory (default: `results`)
- `--resume` — resume into an existing output file, skipping already processed records
- `--limit` — process only first N records (useful for testing)

### Stage 2: Judge responses

Evaluate model responses for refusal using an LLM judge:

```bash
python run_benchmark.py judge --input results/<generate_output>.jsonl
```

Options:
- `--judge-model` — judge model ID (default: `anthropic/claude-sonnet-4.6`)
- `--concurrency` — max parallel requests (default: 15)
- `--limit` — judge only first N records

The judge outputs Refusal Rate — the proportion of prompts for which the model refused to produce harmful content — with breakdowns by toxicity category and jailbreak family.
