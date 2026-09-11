# Training recipe and source disclosure

This document describes the selected original step-3500 run. It is a methods record, not a complete retraining executable. The released runtime and weights support local inference; exact retraining also depends on authorized source data and restricted bookkeeping excluded from this repository.

## Data preparation

Sources are the official HeiCo-FOCUS and LapChole-FOCUS FRAME and SEGMENT distributions. Their released `train.parquet` and `test.parquet` files are pooled, then partitioned at complete-video level into local training and development sets. These are released resource names and are not the hidden platform test-phase references.

The pool contains 28,000 QA records. Local training has 104 videos / 22,400 records; development has 26 videos / 5600 records. Each track contributes 11,200 training and 2800 development records. A deterministic seed-2026 policy protects rare training subtypes, so some rare subtypes may be absent from development. The generic implementation is in `methods/split_policy.py`; actual video identifiers and realized split manifests are excluded.

The planned 20,000-exposure schedule has 17,000 SEGMENT slots and 3000 FRAME slots using `capability_balanced_v2`. The chosen 3500 updates consume 14,000 exposures: 11,900 SEGMENT and 2100 FRAME. Those represent 9649 unique SEGMENT rows plus 2100 unique FRAME rows. Sampling changes frequency, not source labels.

The only explicit label edits are the three answer-text cleanups documented in `annotations/README.md`. They apply only after exact ID and original-answer matches; two affect local training and one development. No extra QA pairs, pseudo-labels, boxes, masks or temporal annotations are generated. Restricted values are not published and no private-delivery exception is claimed.

`annotations/source_provenance.json` provides the original source-file relative names, aggregate counts and SHA-256 values. The schemas in `annotations/schemas/` describe the private preparation records without including any actual QA or video instance.

## Optimization

| Setting | Value |
| --- | --- |
| Foundation revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Precision | BF16 foundation; no weight quantization |
| Initialization | Fresh LoRA adaptation of the public foundation |
| Objective | Answer-token negative log likelihood |
| Optimizer | Fused AdamW; betas 0.9/0.95; epsilon 1e-8 |
| Language/control learning rate | 3e-5 |
| Merger learning rate | 2e-5 |
| Weight decay | 0.01 |
| Scheduler | Cosine across planned 5000 updates |
| Warm-up | 150 updates (3% of the planned run) |
| Microbatch / accumulation | 1 / 4 |
| Gradient clipping | Global norm 1.0 |
| Seed | 2026 |
| Selected optimizer step | 3500 |
| Language LoRA rank / alpha | 16 / 32 |
| Control LoRA rank / alpha | 4 / 8 |
| Merger LoRA rank / alpha | 32 / 64 |
| LoRA dropout | 0.05 |
| Frozen modules | All 27 vision encoder blocks and unadapted foundation parameters |
| Memory handling | Gradient checkpointing and activation offload |

The full saved experiment configuration appears in `configs/experiment.json`. Its `optimizer_steps=5000` describes the planned schedule; the model identity is the separate optimizer-step-3500 checkpoint. Historical memory thresholds are training settings and do not replace the deployment target of H100 SXM5 80 GiB.

The recorded training environment used Python 3.12.13, PyTorch 2.5.1+cu121, Transformers 5.9.0, PEFT 0.19.1, external FlashAttention 2, flash-linear-attention 0.2.2 and causal-conv1d 1.5.0.post8. The selected inference image instead uses PyTorch 2.8.0+cu128, SDPA and native linear-attention fallback. Do not install the training attention extras into the clean public inference environment.

A local Qwen3.5-4B judge was used for evaluation only; it did not provide training labels and is not part of deployment. No distillation, reinforcement learning, PROCEDURE training, external generated VQA or subsequent supplemental checkpoint is part of this release.
