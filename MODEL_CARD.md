# MIRAI-CRD step-3500 model card

## Identity and intended use

- Developer: Team MIRAI.
- Task: surgical video question answering for the ORena FOCUS SEGMENT track.
- Foundation: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Adaptation: original optimizer-step-3500 LoRA checkpoint; all 500 saved tensors are included.
- Trainable parameters: 41,474,048 across 250 adapted linear modules.
- Target inference hardware: one H100 SXM5 with 80 GiB VRAM; BF16 foundation model with the saved adapter, no weight quantization.
- Intended release purpose: public access and local reproduction of the challenge method. This is a research system, without clinical-use validation.

## Architecture and preprocessing

The runtime preserves native-resolution pixels, samples video at 1 FPS, pads the image edge to a multiple of 32 and uses a temporal patch size of 2. Common/residual/delta selection operates after vision block 8 and retains complete groups of four patch tokens. Remaining vision blocks process selected tokens using original spatial/temporal coordinates. The saved configuration specifies 38 visual budget units. Generation is greedy with at most 32 new tokens and thinking disabled.

The adapter includes language attention/linear-attention, control-projection and visual-merger LoRA tensors. Language LoRA uses rank 16/alpha 32, control `in_proj_a`/`in_proj_b` rank 4/alpha 8, and the two merger projections rank 32/alpha 64; dropout is 0.05. The vision encoder blocks remain frozen. The complete loader in the accompanying code repository must be used; merging only a subset of tensors or omitting the CRD runtime is not equivalent.

## Training resources

Training uses the released FRAME and SEGMENT resources from HeiCo-FOCUS and LapChole-FOCUS, with a video-disjoint local train/development partition and deterministic sampling. Source distributions' `train.parquet` and `test.parquet` names refer to released resources, not hidden challenge evaluation references. Training reached 3500 optimizer updates and 14,000 example exposures. Only this checkpoint is released; later experiments are excluded.

Three LapChole answer cleanups removed trailing editorial notes without changing the preceding answer text (two local-training records and one development record). Restricted before/after values are not included. No private-publication exception has been approved; organizer clarification about these edits remains outstanding. See the accompanying repository's annotation disclosure, training recipe and source provenance.

## Assets and loading

The model archive contains `adapter/`, `processor/`, `run_manifest.json`, this card and licensing notices, under a `model/` directory. The public run manifest retains the exact inference `config.experiment` and omits private training records. The tensor and processor bytes are unchanged from the selected container. SHA-256 values are published in the accompanying repository's `RELEASE_MANIFEST.json`.

Download the exact public foundation revision and follow that repository's `README.md` using `scripts/restore_model.py`, `scripts/download_base_model.py` and `scripts/run_inference.py`. No access to the training dataset is required merely to load the model and run it on an independently supplied compatible input.

## Evaluation and limitations

No hidden-test results or unpublished development-score tables are reported in this release. H100 is the deployment target, not a claim that public-image GPU inference was independently tested during preparation. The original container has historical integrity/reload evidence; public packaging checks verify preservation and restoration of its model files. The reconstructed public Dockerfile is a separate, unbuilt recipe.

Responses may be incorrect, and performance can vary with procedure, object visibility, clip duration and memory load. The runtime includes fallback behavior, so inspect `model_success`, `fallback` and `deadline_skip` in the batch summary when validating inference. No claims are made for clinical decisions or domains beyond the challenge task.

## Terms

The foundation is subject to its upstream Apache-2.0 license, included with the archive. Adapter access and reproduction terms are described in `MODEL_LICENSE.md`. MIT applies to MIRAI-owned code in the accompanying repository and does not grant source-data or annotation rights.
