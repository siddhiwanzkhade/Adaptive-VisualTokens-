# Adaptive-VisualTokens


Question-aware adaptive visual-token selection for LLaVA-1.5-7B that reduces the number of CLIP image tokens sent to Vicuna(LLM) while preserving useful VQA quality.

## Problem

LLaVA converts every image into 576 visual tokens and sends all of them to Vicuna, even when a question only depends on a small part of the image.

This increases memory usage and multimodal prefill computation unnecessarily.


## Tech Stack and Setup

- **Model:** `llava-hf/llava-1.5-7b-hf`
- **Architecture:** CLIP ViT-L/14-336 + Vicuna-7B
- **Token compression:** question-aware adaptive selection before Vicuna
- **Frameworks:** Python, PyTorch, Hugging Face Transformers
- **Quantization:** bitsandbytes 4-bit NF4
- **Fine-tuning:** PEFT / QLoRA
- **Dataset:** VQAv2 validation subset
- **Data tools:** Hugging Face Datasets, NumPy, Pandas, Pillow
- **Hardware:** NVIDIA A100 on Google Colab

The reported VQA score uses an approximate normalized substring-based soft score, not the official VQAv2 evaluation metric.

## Approach

```text
Image ──▶ CLIP vision encoder ──▶ 576 projected visual tokens
                                        │
Question ──▶ question embedding ───────┤
                                        ▼
                          question-aware token scoring
                       0.7 × relevance + 0.3 × visual norm
                                        │
                                        ▼
                    adaptive cumulative-coverage selection
                              bounded to [32, 512]
                                        │
                                        ▼
                     restore selected tokens to their
                          original spatial order
                                        │
                                        ▼
                         reduced visual-token sequence
                                        │
                                        ▼
                     Vicuna generates the first answer
                                        │
                                        ▼
                    confidence and entropy evaluation
                              │                   │
                     confidence high        confidence low
                              │                   │
                              │                   ▼
                              │          increase coverage threshold
                              │                   │
                              │                   ▼
                              │          reselect more visual tokens
                              │                   │
                              │                   ▼
                              │          regenerate answer once
                              │                   │
                              └───────────┬───────┘
                                          ▼
                         final answer + tokens used

Tokens are selected using a cumulative relevance threshold instead of a fixed `K`.

Selected tokens are restored to their original spatial order before being passed to Vicuna.


## Experiments

### Experiment 1 — Fine-Tuning Vicuna for Compressed Visual Inputs

QLoRA fine-tuning was applied to help Vicuna adapt to the reduced visual-token sequence.

```text
Frozen compressed model: 49%
QLoRA compressed model:  55%
```

The adapter trained approximately 8.4M parameters, or 0.12% of the model.

QLoRA recovered some quality, but it did not improve inference speed. Its apparent latency improvement came from generating shorter answers rather than faster decoding.

### Experiment 2 — Batched Inference Optimization with Adaptive Visual Tokens

Compared full-token inference with adaptive-token inference at batch sizes 1, 2, and 4 using the same model and generation settings.

| Batch | Adaptive Tokens | Token Reduction | Latency Change | Throughput Change | Memory Change |
|---:|---:|---:|---:|---:|---:|
| 1 | 165.0 | 71.35% | 2.2% slower | 2.1% lower | 2.86% lower |
| 2 | 170.0 | 70.49% | 4.03% lower | 4.20% higher | 4.42% lower |
| 4 | 174.75 | 69.66% | 9.72% lower | 10.77% higher | 8.59% lower |

At batch size 4, adaptive token selection achieved:

- 69.7% fewer visual tokens
- 9.7% lower latency
- 10.8% higher throughput
- 8.6% lower peak GPU memory

The results suggest that visual-token compression becomes more useful as batch size increases.


Then keep the technical section separately:

## Results

### Training-Free Baseline — VQAv2, n=100

| Metric | Result |
|---|---:|
| Visual-token reduction | 61–68% |
| Tokens processed | 576 → approximately 183–222 |
| Full-token VQA score | 0.690 |
| Compressed VQA score | 0.527 |
| Quality retained | approximately 76% |
| Peak memory saved | approximately 5.4% |
| Batch-1 latency | no meaningful improvement |
| Batch-4 latency | 9.7% lower |
| Batch-4 throughput | 10.8% higher |
| Batch-4 peak memory | 8.6% lower |

The selector removes most visual tokens, but some VQA accuracy is lost.

At batch size 1, compression does not improve latency. At batch size 4, the same method reduces latency, increases throughput, and lowers peak GPU memory.

## Known Limitations

- The compressed baseline loses some VQA quality.
- Confidence fallback cannot detect every confidently incorrect answer.
- Batched results are preliminary and should be repeated across more batches.
- Published token-reduction methods have not yet been evaluated at matched compression ratios.


