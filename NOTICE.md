# Attribution and license scope

MIRAI-CRD is developed by Team MIRAI for the ORena FOCUS SEGMENT track. MIRAI-owned code and documentation are offered under the root MIT license, except where a file or component states another applicable upstream license. Dataset and model terms are described separately in `MODEL_LICENSE.md` and `annotations/README.md`.

The project uses the challenge interface and workflows associated with the following upstream projects. Their license and notice texts are retained to preserve attribution for inherited or adapted material; this does not relicense upstream components as MIRAI-owned MIT code.

| Upstream | Terms retained here |
| --- | --- |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) | `licenses/Qwen3.5-9B_APACHE-2.0.txt` |
| [IMSY-DKFZ/orena-focus](https://github.com/IMSY-DKFZ/orena-focus/tree/7b7e5c537518e8a65f8d70b89dedc3b4518a856a) | `licenses/orena-focus_MIT.txt` |
| [IMSY-DKFZ/orena-focus-submission-template](https://github.com/IMSY-DKFZ/orena-focus-submission-template) | `licenses/submission-template_APACHE-2.0.txt`, `licenses/submission-template_NOTICE.txt` |

The Qwen foundation is adapted through the separately distributed MIRAI LoRA weights and the CRD inference implementation. Foundation-model files are obtained directly from the pinned public upstream revision. Retained processor files come from the selected checkpoint.

Public packaging generalizes laboratory filesystem defaults in `runtime/patch_selection/config.py`, minimizes the checkpoint's metadata to its exact inference configuration, and adds installation, restoration and launch utilities. Adapter tensors, processor bytes and the inference algorithms are preserved. `RELEASE_PROVENANCE.json` records the original and public runtime hashes and these changes.

PyTorch, Transformers, PEFT, Accelerate, NumPy, Pillow and Decord are installed as dependencies and retain their own licenses. HeiCo-FOCUS and LapChole-FOCUS data are not bundled. The repository does not grant permission to redistribute challenge datasets or annotations.
