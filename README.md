# Adaptive-VisualTokens

Question-aware adaptive visual-token selection for LLaVA-1.5-7B, reducing the number of image tokens processed by the language model while preserving useful VQA quality.

## Problem

LLaVA converts every image into 576 visual tokens and sends all of them to Vicuna, even when a question only depends on a small part of the image.

This increases memory usage and multimodal prefill computation unnecessarily.

## Approach

```text
Image ──▶ CLIP ──▶ 576 visual tokens
                         │
Question ──▶ embedding ──┤
                         ▼
              question-aware scoring
       0.7 × relevance + 0.3 × visual norm
                         │
                         ▼
          adaptive token selection [32, 512]
                         │
                         ▼
             Vicuna with compressed input
                         │
                         ▼
          confidence-based fallback once
```

Tokens are selected using a cumulative relevance threshold instead of a fixed `K`.

Selected tokens are restored to their original spatial order before being passed to Vicuna.

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

The selector removes most visual tokens, but some VQA accuracy is lost. The confidence fallback reruns uncertain samples with a higher token-coverage threshold.

## Experiments

### Experiment 1 — QLoRA Quality Recovery

QLoRA was used to adapt Vicuna to compressed visual inputs.

```text
Frozen compressed model: 49%
QLoRA compressed model:  55%
```

The adapter trained approximately 8.4M parameters, or 0.12% of the model.

QLoRA recovered some quality, but it did not improve inference speed. Its apparent latency improvement came from generating shorter answers rather than faster decoding.

### Experiment 2 — Batched Inference

Full-token and adaptive-token inference were compared at batch sizes 1, 2, and 4.

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

## Setup

```text
Model: llava-hf/llava-1.5-7b-hf
Vision encoder: CLIP ViT-L/14, 336px
Language model: Vicuna-7B
Quantization: 4-bit NF4
Dataset: VQAv2 validation subset
Hardware: single A100 on Google Colab
```

The reported VQA score uses an approximate normalized substring-based soft score, not the official VQAv2 evaluation metric.

## Repository Structure

```text
Adaptive-VisualTokens/
├── run_swiftvlm.ipynb
├── README.md
└── results/
```

## Known Limitations

- The compressed baseline loses some VQA quality.
- Confidence fallback cannot detect every confidently incorrect answer.
- Batched results are preliminary and should be repeated across more batches.
- Published token-reduction methods have not yet been evaluated at matched compression ratios.

## Stack

Python, PyTorch, Transformers, PEFT, bitsandbytes, Hugging Face Datasets, NumPy, Pandas, and Matplotlib.
