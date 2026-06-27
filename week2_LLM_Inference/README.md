# Week 2: LLM Inference Foundations

This folder covers the input side of LLM inference: how raw text becomes tokens, how token count affects context fit, and how prompt size influences latency.

## What We Built

1. Tokenizer comparison across open-source tokenizers
2. Context-window fit analysis
3. Prompt packing and truncation behavior
4. Prefill latency estimation
5. Decode latency estimation
6. Full inference simulation
7. Real local benchmark with Ollama

## Core Files

- `src/tokenizer_utils.py`: load tokenizers, inspect token pieces, compare token counts
- `src/context_analyzer.py`: check whether prompts fit within a context window
- `src/prompt_packer.py`: pack prompt sections by priority under token limits
- `src/prefill_estimator.py`: estimate time-to-first-token from prompt size
- `src/decode_estimator.py`: estimate decode time from output length and tokens/sec
- `src/inference_analyzer.py`: combine tokenization, context, prefill, and decode into one simulation
- `src/ollama_real_benchmark.py`: run real local measurements against Ollama models

## Example Inputs

- `examples/sample_texts.txt`
- `examples/tokenization_cases.json`
- `examples/context_examples.json`
- `examples/prompt_sections.json`
- `examples/full_prompt_example.txt`

## Outputs

- `outputs/tokenization_behavior_report.csv`
- `outputs/context_window_report.csv`
- `outputs/prompt_packing_report.csv`
- `outputs/prefill_estimation_report.csv`
- `outputs/decode_estimation_report.csv`
- `outputs/full_inference_analysis.csv`
- `outputs/full_inference_analysis_compare.csv`
- `outputs/real_benchmark_summary.csv`
- `outputs/benchmark_runs/`: archived real benchmark runs

## Suggested Learning Order

1. Read `src/tokenizer_utils.py`
2. Read `src/context_analyzer.py`
3. Read `src/prompt_packer.py`
4. Read `src/prefill_estimator.py`
5. Read `src/decode_estimator.py`
6. Read `src/inference_analyzer.py`
7. Read `src/ollama_real_benchmark.py`

Then open the matching notebooks in `notebooks/` to see the same ideas step by step.

## Simulation vs Real Benchmark

- `prefill_estimator.py`, `decode_estimator.py`, and `inference_analyzer.py` are estimation tools
- `ollama_real_benchmark.py` measures actual local latency using downloaded models

Both are useful:
- simulation builds intuition quickly
- real benchmarking shows actual machine and model behavior

## Run The Real Benchmark

Start Ollama in another terminal if needed:

```bash
ollama serve
```

Run the benchmark from the repo root:

```bash
week2_LLM_Inference/week2_llm_inf/bin/python week2_LLM_Inference/src/ollama_real_benchmark.py \
  --prompt-file week2_LLM_Inference/examples/full_prompt_example.txt \
  --models qwen3:4b mistral:instruct gemma3:4b \
  --num-predict 512 \
  --num-runs 50 \
  --warmup-runs 3 \
  --temperature 0 \
  --output-tag week2_final_512tok
```

## Docker Workflow

The repository now includes a Docker-based benchmark path for real local inference.

Start the Ollama server container:

```bash
cd week2_LLM_Inference
docker compose up -d ollama
```

Pull the benchmark models into the Ollama volume:

```bash
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama pull mistral:instruct
docker compose exec ollama ollama pull gemma3:4b
```

Run the benchmark container:

```bash
docker compose run --rm benchmark \
  --prompt-file examples/full_prompt_example.txt \
  --models qwen3:4b mistral:instruct gemma3:4b \
  --num-predict 512 \
  --num-runs 50 \
  --warmup-runs 3 \
  --temperature 0 \
  --output-tag docker_512tok_run
```

The benchmark outputs will be written into `outputs/` on the host machine.

## Key Takeaways

- Token count changes by tokenizer even for the same text
- Context limits are really token limits, not word limits
- Prompt structure affects what survives truncation
- Longer prompts raise prefill cost
- Longer outputs raise decode time
- Real latency depends on the actual model and hardware, not just token counts
