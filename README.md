# LLM Inference Cost & Latency Analyzer

Hands-on LLM infrastructure project focused on what happens before and during inference: tokenization, context-window fit, prompt packing, latency estimation, and real local benchmarking.

## Current Scope

This repository currently contains the Week 2 project:

- `week2_LLM_Inference/`

That module covers:

- tokenizer comparison across open-source models
- context-window analysis
- prompt packing and truncation
- prefill estimation
- decode estimation
- full inference simulation
- real local benchmarking with Ollama

## Project Structure

```text
week2_LLM_Inference/
├── examples/     # sample prompts and structured input cases
├── notebooks/    # step-by-step learning notebooks
├── outputs/      # generated CSV reports and figures
├── src/          # runnable Python scripts
└── README.md     # week-specific guide
```

## Quick Start

Create or activate the Week 2 environment, then install dependencies:

```bash
cd week2_LLM_Inference
python3 -m venv week2_llm_inf
source week2_llm_inf/bin/activate
pip install -r requirements.txt
```

Run the real benchmark from the repo root:

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

If Ollama is not already running:

```bash
ollama serve
```

## Main Files

- `week2_LLM_Inference/src/tokenizer_utils.py`
- `week2_LLM_Inference/src/context_analyzer.py`
- `week2_LLM_Inference/src/prompt_packer.py`
- `week2_LLM_Inference/src/prefill_estimator.py`
- `week2_LLM_Inference/src/decode_estimator.py`
- `week2_LLM_Inference/src/inference_analyzer.py`
- `week2_LLM_Inference/src/ollama_real_benchmark.py`

## Outputs

Generated outputs include:

- tokenization reports
- context-window reports
- prompt packing reports
- prefill and decode estimation reports
- full inference analysis
- real benchmark summaries
- figures for visual comparison

See `week2_LLM_Inference/outputs/` for the latest generated artifacts.

## Notes For GitHub

- local virtual environments are ignored
- archived benchmark reruns are ignored
- current generated outputs remain available for reference

## Next Nice-To-Haves

- add a repository license
- add a few screenshots of the figures in the README
- initialize git and push to GitHub
