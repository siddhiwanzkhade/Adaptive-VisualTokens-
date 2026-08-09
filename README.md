# SwiftVLM : Adaptive-VisualTokens


Question-aware adaptive visual-token selection for LLaVA-1.5-7B that reduces the number of CLIP image tokens sent to Vicuna(LLM) while preserving useful VQA(Visual Question Answering) quality.

## Problem

LLaVA converts every image into 576 visual tokens and sends all of them to Vicuna, even when a question only depends on a small part of the image.

This increases memory usage and multimodal prefill computation unnecessarily.

## Idea

Apply question-aware adaptive token pruning between CLIP and Vicuna, so only the visual embeddings most relevant to the question are forwarded to the language model instead of all 576 tokens.

This reduces the multimodal sequence length and inference cost without using a fixed token budget.

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

```

Tokens are selected using a cumulative relevance threshold instead of a fixed `K`.

The selected tokens are restored to their original spatial order before being passed to Vicuna.


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

Compared full-token inference with adaptive-token inference at batch sizes 1, 2, and 4 , 8, 16 using the same model and generation settings.

| Batch | Adaptive Tokens | Token Reduction | Latency Change | Throughput Change | Memory Change |
|---:|---:|---:|---:|---:|---:|
| 1 | 165.0 | 71.35% | 2.2% slower | 2.1% lower | 2.86% lower |
| 2 | 170.0 | 70.49% | 4.03% lower | 4.20% higher | 4.42% lower |
| 4 | 174.75 | 69.66% | 9.72% lower | 10.77% higher | 8.59% lower |
| 8	| 161.125	| 72.03%	| 12.73% lower | 14.59% higher	|15.19% lower |
|16	| 165.5625 | 71.26%	| 22.98% lower	| 29.83% higher	| 25.25% lower |

At batch size 16, adaptive token selection achieved:
- 71.3% fewer visual tokens
- 23.0% lower latency
- 29.8% higher throughput
- 25.3% lower peak GPU memory
Gains scaled consistently with batch size — negligible at batch 1, growing steadily through batch 16 — indicating the model shifts from memory-bandwidth-bound toward compute-bound as batch size increases, which is where token compression pays off.

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
| Batch-4 latency |	9.7% lower |
| Batch-8 latency	| 12.7% lower |
| Batch-16 latency |	23.0% lower |
| Batch-16 throughput	| 29.8% higher |
| Batch-16 peak memory| 25.3% lower |


The selector removes most visual tokens, but some VQA accuracy is lost.
At batch size 1, compression does not improve latency. As batch size increases, the same method reduces latency, increases throughput, and lowers peak GPU memory — with gains growing from batch 2 through batch 16, reaching a 23.0% latency reduction, 29.8% throughput improvement, and 25.3% memory reduction at batch size 16.

### Kernal level GPU Profiling — Locating the Real Bottleneck

To understand why compression didn't improve batch-size-1 latency, inference was profiled at the GPU kernel level using NVIDIA Nsight Systems (NSYS).

The kernel-time breakdown showed that `kQuantizeBlockwiseSmall` (the 4-bit weight dequantization kernel) accounted for 54–67% of total GPU time — far more than the actual matrix-multiplication (GEMM) kernels used for computation. This kernel count did not change between the baseline and adaptive-token runs, since it scales with model weights, not input size.

This confirmed that at batch size 1, the model is memory-bandwidth bound rather than compute bound — reducing visual tokens shrinks GEMM input, but GEMM was never the dominant cost. This is why token compression alone did not improve single-request latency, and why its benefit only appeared once batching shifted the bottleneck toward compute.

### MAIN CONCLUSION:

Visual-token compression for VLM inference isn't universally beneficial — its value depends entirely on the serving regime. At low batch sizes, the model is memory-bandwidth bound, so reducing tokens barely helps. At higher batch sizes, the bottleneck shifts toward compute, and the same compression method delivers real, scaling gains — up to 23% lower latency, 30% higher throughput, and 25% lower memory at batch size 16. The right question isn't "does token compression work," it's "under what conditions does it work" — and this project answers that with profiling evidence, not assumption.

### MAIN RESULT : 
~71% fewer visual tokens → ~30% higher throughput at batch size 16, with ~76% VQA accuracy retained.
Throughput is what production inference systems actually optimize for — this shows the token reduction directly drives real serving gains, not just a smaller input.

## Known Limitations

- The compressed baseline loses some VQA quality.
- Confidence fallback cannot detect every confidently incorrect answer.
- Batched results tested up to batch size 16; not yet validated on larger production batch sizes or real serving frameworks (e.g. vLLM).
- Published token-reduction methods have not yet been evaluated at matched compression ratios.


