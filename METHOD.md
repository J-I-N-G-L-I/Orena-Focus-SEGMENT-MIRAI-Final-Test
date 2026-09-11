# Method: native-resolution common/residual/delta token selection

## Submitted system

MIRAI-CRD uses the original step-3500 Qwen3.5-9B LoRA checkpoint for surgical video question answering. The visual backbone is frozen; language, control and merger projections are adapted. The released inference runtime implements the full method, including token selection, position handling, model loading, prompts and answer generation.

The foundation-model revision and all inference hyperparameters are fixed in `RELEASE_MANIFEST.json` and `configs/experiment.json`. Later supplemental-training and alternate temporal-prompt experiments are excluded.

## Video representation

The runtime samples clips at 1 FPS, includes the last frame and duplicates the final frame when needed to satisfy temporal patching. It preserves the original pixel dimensions and pads right/bottom edges to multiples of 32. No spatial resizing or crop is applied under `resolution_mode=native_pad32`.

Qwen's temporal patch size is 2 and spatial merge size is 2. Four spatial patch tokens form one complete merge group. The vision encoder processes blocks 0 through 8 densely; descriptors for selection are computed from these intermediate activations. Mean descriptors are used for scoring only: retained original patch tokens, rather than averaged descriptors, enter the downstream blocks and merger.

## CRD selection

The selector estimates common structure using an SVD-based subspace with a 0.90 energy threshold. Segmentation uses `min_units=2` and `max_units=16`, with a 0.70 common-energy threshold; the final segment may be shorter. It retains temporal coverage anchors and allocates the remaining budget approximately 40%/30%/30% across common, residual and delta criteria. See `runtime/patch_selection/selection.py` for exact integer allocation, ranking and deterministic tie handling.

All selected merge groups retain their original spatial/temporal coordinates. Blocks 9 through 26 process the sparse representation, which is passed through the adapted visual merger and into the language model. Original position information is retained rather than treating selected groups as a new dense crop.

The actual checkpoint uses **38 visual budget units**. The `height=288`, `width=512` and `visual_token_budget=5472` entries are retained historical configuration fields; native-resolution budgeting derives the effective budget from the padded grid. For example, a 1280×720 frame becomes 1280×736 with a 34,960 merged-token budget; 960×540 becomes 960×544 with a 19,380 budget. The saved `native_pad32` mode and runtime resolution code determine behavior.

## Adaptation and optimization

LoRA targets 250 linear modules and saves 500 tensors (41,474,048 parameters). Language projections use rank 16/alpha 32, control projections use rank 4/alpha 8, and visual-merger projections use rank 32/alpha 64. LoRA dropout is 0.05; bias training, DoRA and rank-stabilized LoRA are not used. Vision encoder blocks are frozen.

Optimization uses answer-token negative log likelihood, a microbatch of 1, gradient accumulation of 4, and seed 2026. The original run planned 5000 optimizer steps; this release selects step 3500. Detailed settings and data treatment are in `TRAINING.md`.

## Inference and challenge interface

The prompt combines the request, procedure-clock window, visual-time table and supplied foreign-object definitions. The processor's chat template is fixed to the checkpoint; thinking is disabled. Greedy decoding generates at most 32 new tokens. `runtime/patch_selection/prompting.py` and `runtime/submission/inference.py` contain the exact prompt and entry-point behavior.

The selected container uses PyTorch 2.8.0+cu128 with SDPA and the native linear-attention fallback, under the `legacy-fallback` profile. This differs from the faster attention extensions used in the training environment. The public installer uses the corresponding direct dependency versions without those optional extensions.

The entry point reads the official request/FO-definition/video inputs and writes `answer.json`. It applies batch-time limits and may emit fallback answers following errors or deadline skips. Reported successful process completion must therefore be checked against the batch summary counters before calling a run a model-inference success.

## Reproduction boundary

The complete trained adapter plus the fixed public foundation revision and runtime can reconstruct this inference method without access to private training records. Model files and processor bytes are preserved from the selected container. The metadata file is reduced to the exact experiment configuration; unused laboratory path defaults in one source file are generalized. Hashes and changes are listed in `RELEASE_PROVENANCE.json`.

The repository does not reproduce the original container bytes or provide a complete training pipeline. Exact data rows and restricted corrections are excluded under the source-data conditions. The public model archive restore is validated separately from GPU inference. No hidden-test or unpublished development scores are presented here.
