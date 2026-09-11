# MIRAI-CRD: SEGMENT challenge model

Public inference release for Team **MIRAI** in the [ORena FOCUS SEGMENT track](https://segment.orena-focus-challenge.org/). This release contains the original **optimizer-step-3500** adapter, its processor, exact inference configuration and the inference code used by the selected container.

The method adapts Qwen3.5-9B using LoRA and reduces native-resolution video tokens through common/residual/delta (CRD) selection inside the vision encoder. Target deployment is one **NVIDIA H100 SXM5, 80 GiB**.

## Get the model

Open this repository's **Releases → v1.0.0** and download both assets:

- `mirai-segment-step3500-model.tar.gz`
- `mirai-segment-step3500-model.tar.gz.sha256`

The model archive is separate from Git history. GitHub's automatically generated **Source code** archives do not contain the weights. The archive contains every trained adapter tensor, the processor and the inference configuration. Download the public foundation model separately using the pinned downloader below.

| Component | Exact release |
| --- | --- |
| Foundation model | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Foundation revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Checkpoint | Original step 3500; no subsequent supplemental training |
| Adapter | 500 tensors; 41,474,048 trainable parameters |
| Video processing | 1 FPS; native pixels; edge padding to multiples of 32 |
| Selection | Layer 8 CRD; **38** visual budget units from the saved configuration |
| Generation | Greedy; at most 32 new tokens |
| Submission profile | `legacy-fallback`, PyTorch SDPA |

See [MODEL_CARD.md](MODEL_CARD.md), [METHOD.md](METHOD.md), [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) and [RELEASE_PROVENANCE.json](RELEASE_PROVENANCE.json).

## Local inference

Use Linux with Python 3.11 and a CUDA-capable PyTorch environment suitable for an H100. Create a clean environment without extra FlashAttention/FLA packages: `legacy-fallback` does not itself disable an optional FLA kernel already installed in the environment. Run the following from the repository root. Installation and foundation-model download require internet access; inference is offline. Provision space for the approximately 18 GB foundation model, caches and, if used, Docker layers.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# Save the Release model asset in downloads/ first.
python scripts/restore_model.py downloads/mirai-segment-step3500-model.tar.gz
python scripts/download_base_model.py --output "$PWD/resources/base_qwen35_9b"

python scripts/run_inference.py \
  --input /absolute/path/to/input \
  --output /absolute/path/to/output
```

`restore_model.py` checks the archive SHA-256 and every file against `RELEASE_MANIFEST.json` before installing the files under `model/`. The downloader fixes the foundation revision rather than tracking `main`. `run_inference.py` loads the complete adapter through the released runtime; a plain foundation-model `generate()` call does not reproduce CRD selection.

For a base model already downloaded elsewhere, pass `--base-model /absolute/path/to/base`. For adapter files installed elsewhere, pass `--model /absolute/path/to/model`.

The official-format input layout is:

```text
input/
  request.json
  FO_definitions.json
  plain/
    <qID>.mp4
```

`request.json` is a list of request objects containing `qID`, `videoID`, `start_time`, `end_time`, `procedure_type` and `question`. The output is `answer.json`, a list of objects containing `qID`, `content` and `latency`. Use inputs obtained through the official challenge tooling and data-access process. Source videos and annotation examples are not redistributed here. See the [official submission template](https://github.com/IMSY-DKFZ/orena-focus-submission-template).

**Verify real model execution:** the frozen entry point can write fallback answers after an error and still exit successfully. Inspect its final `[batch-summary]`: `model_success` must equal the request count, `fallback=0`, and `deadline_skip=0`. An existing `answer.json` or exit code 0 alone is insufficient.

## Docker reconstruction

The provided Dockerfile is a public reconstruction recipe. It produces a new image; it is not the original submission image byte for byte. Restore `model/` and download the foundation model into `resources/base_qwen35_9b/` first, as above.

```bash
docker build -t mirai-segment:step3500-public .
mkdir -p /absolute/path/to/output
# Ensure this output directory is writable by container UID 999.
docker run --rm --gpus all --network none --shm-size=8g \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  mirai-segment:step3500-public
```

Direct dependencies are pinned in `requirements.txt`; it is not a complete transitive lock of the original image. This public reconstruction has not been built or GPU-tested during release preparation. The original image passed archive-integrity and Docker-reload checks; its locally recorded interface smoke test used CPU fallback and does not establish H100 model-inference success. The team performed separate laboratory testing of its selected container; that statement is not represented here as an independently reproduced public-image test.

## Data, annotations and reproduction scope

Training used organizer-provided [HeiCo-FOCUS](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and [LapChole-FOCUS](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) FRAME and SEGMENT resources. No PROCEDURE training, external generated VQA, distillation or pseudo-label supervision was used for this checkpoint.

Three LapChole answers had trailing editorial notes removed: two in local training and one in local development. The exact restricted IDs and answers are excluded. [The annotation disclosure](annotations/README.md) explains the edits and the unresolved organizer clarification concerning their release. Public schemas and counts are not claimed to substitute for publication of any annotations the organizers determine must be released.

This is an inference release with a detailed [training recipe](TRAINING.md), configuration and generic split-policy code. Full training orchestration, restricted data tables, private split manifests, optimizer states and experimental history are not included. Retraining the exact data realization requires the authorized source data and restricted bookkeeping; this release does not claim one-command exact retraining. Local inference requires only the public model assets, the pinned foundation model and user-supplied inputs.

## License and acknowledgements

MIRAI-owned code is licensed under [MIT](LICENSE). Retained upstream notices and component terms are listed in [NOTICE.md](NOTICE.md). Foundation-model and adapter terms are explained separately in [MODEL_LICENSE.md](MODEL_LICENSE.md); the code license grants no dataset or annotation rights.

We acknowledge the Qwen team, the ORena FOCUS organizers and dataset contributors, and the maintainers of PyTorch, Transformers and PEFT. This repository is a challenge model and reproducibility release; it does not report hidden-test results or claim prize eligibility has been confirmed.
