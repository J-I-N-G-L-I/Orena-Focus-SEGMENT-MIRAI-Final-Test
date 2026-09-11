# Model terms and scope

The repository-root MIT license covers MIRAI-owned code. It does **not** automatically license the foundation-model weights, the trained adapter, the processor or the source datasets.

## Foundation and processor

The foundation model is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a). Its [upstream license at the pinned revision](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE) is Apache-2.0. That text is retained as `licenses/Qwen3.5-9B_APACHE-2.0.txt` in the code repository and `LICENSE_QWEN.txt` in the model archive. Preserve applicable upstream notices when redistributing upstream materials.

## MIRAI adapter

This release publicly distributes Team MIRAI's step-3500 adapter for downloading, loading and reproducing the ORena FOCUS SEGMENT submission. The adapter is a fine-tuned modification of the named foundation model; it is not an official Qwen release. All trained adapter tensors and the runtime needed to apply them are provided.

This document does not assert an unrestricted commercial license for the MIRAI adapter or label all weights as MIT. For use beyond challenge reproduction, obtain applicable permissions from the relevant rights holders and check the current terms of the training resources. Public availability and an unrestricted software license are different matters.

## Training resources

The dataset providers retain their respective data and annotation rights. Obtain HeiCo-FOCUS and LapChole-FOCUS from their original providers under their current access and use conditions:

- [HeiCo-FOCUS dataset card](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa)
- [LapChole-FOCUS data-use agreement](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)

No source dataset, answer table, restricted answer correction or video is licensed or redistributed by this release. These dataset conditions are not represented here as automatically determining the license of every trained weight. The LapChole agreement distinguishes model use from redistribution of data and underlying annotations.
