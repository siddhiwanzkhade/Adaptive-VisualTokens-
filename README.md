# Adaptive-VisualTokens

Question-aware adaptive visual-token selection for LLaVA-1.5-7B, so the language model only processes the image tokens a given question actually needs.

## Problem

LLaVA encodes every image into 576 fixed patch tokens (CLIP ViT-L/14, 336px, 24×24 grid) and passes all 576 into Vicuna regardless of the question. A question like *"where is the ball?"* only needs a small region of the image, but the model pays full prefill cost for the entire frame every time.

## Approach

```text
Image ──▶ CLIP vision encoder ──▶ 576 projected visual tokens
                                        │
Question ──▶ question embedding ────────┤
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
