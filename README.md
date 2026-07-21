# Adaptive-VisualTokens

Question-aware adaptive visual-token selection for LLaVA-1.5-7B, so the language model only processes the image tokens a given question actually needs.

## Problem

LLaVA encodes every image into 576 fixed patch tokens (CLIP ViT-L/14, 336px, 24×24 grid) and passes all 576 into Vicuna regardless of the question. A question like *"where is the ball?"* only needs a small region of the image, but the model pays full prefill cost for the entire frame every time.

## Approach

```
Image ──▶ CLIP vision encoder ──▶ 576 projected visual tokens
                                        │
Question ──▶ question embedding ───────┤
                                        ▼
                          question-aware token scorer
                       (0.7 × relevance + 0.3 × visual norm)
                                        │
                                        ▼
                     select top-K tokens to hit a relevance-
                     coverage threshold (bounded to [32, 512])
                                        │
                                        ▼
                              Vicuna-7B (compressed input)
                                        │
                                        ▼
                    confidence check (avg token prob / entropy)
                low confidence ──▶ raise coverage, reselect, regenerate once
```

Selected tokens keep their original spatial order before being passed to the LLM, so positional structure is preserved.

## Results (training-free baseline, n=100, VQAv2)

| Metric | Value |
|---|---|
| Token reduction | 61–68% (576 → ~183–222 avg) |
| VQA soft-score retained | ~76% of full-token baseline (0.69 → 0.527) |
| Fallback triggered | ~49% of samples |
| Peak GPU memory saved | ~5.4% |
| Single-request latency | **no measurable speedup** |

The last row is the most important finding. Isolating prefill from decode (rather than timing `generate()` end-to-end, which hides the effect) showed prefill time barely moves with fewer tokens. At batch size 1, inference is memory-bandwidth bound, not compute bound — the GPU spends its time loading model weights, not doing math on tokens, so shrinking the input doesn't shrink prefill wall-clock time here. Token reduction pays off in memory, not single-request speed; the hypothesis that it pays off in **throughput under batched serving** (where compute becomes the bottleneck again) is stated but not yet tested.

## QLoRA experiments (adapting Vicuna to compressed input)

**Experiment 1 — fine-tune on original VQAv2 answers.** Recovered some accuracy on compressed input (49% → 55% on held-out samples) but reproduced VQAv2's one-word answer style (`"yes"`, `"2"`, `"red"`), and a length-matched benchmark showed QLoRA decode is actually *slower* per token than frozen Vicuna — the earlier appearance of a speedup was an artifact of QLoRA's shorter outputs, not a real efficiency gain. Retired in favor of Experiment 2.

**Experiment 2 — SwiftVLM-ConciseQA (in progress).** Retraining the same LoRA setup on short, complete-sentence targets (e.g. `"There are 2 visible in the image."` instead of `"2"`) built from the VQAv2 answer, to test whether the model can stay concise without collapsing to single-word outputs. Training is running; evaluation against the frozen and Experiment-1 checkpoints is not complete yet.

## Setup

- `llava-hf/llava-1.5-7b-hf`, 4-bit NF4 quantization (bitsandbytes)
- QLoRA on Vicuna's `q_proj/k_proj/v_proj/o_proj` (8.4M trainable params, 0.12% of total)
- Eval: 100-sample VQAv2 validation subset (seeded), scored with an approximate soft-score (substring match against normalized human answers — not the official VQAv2 metric)
- Hardware: single A100 (Colab)

## Known limitations

- Scoring approximation may undercount numeric answers (e.g. `"5"` vs `"five"`) — flagged for a fix, not yet corrected in the numbers above.
- No comparison yet against published token-reduction methods (FastV, PruMerge, VTW) at matched reduction ratios.
- Batched-throughput testing (the regime where speedup is actually expected) is not yet implemented.

## Stack

Python, PyTorch, Transformers, PEFT (QLoRA), bitsandbytes, Hugging Face Datasets (VQAv2, streaming)
