# Training data and annotation disclosure

The submitted model is MIRAI's original step-3500 checkpoint. It uses the organizer-provided [HeiCo-FOCUS](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and [LapChole-FOCUS](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) FRAME and SEGMENT resources. Both distributed source files named `train.parquet` and `test.parquet` were pooled before constructing a new video-disjoint local train/development split. These filenames refer to released development resources, not hidden challenge test-phase references.

The only explicit answer edits are three exact-match LapChole SEGMENT cleanups. For each, a trailing editorial note beginning with `Errors:` was removed, preserving the preceding answer text. Application required both the complete internal record ID and the full original answer to match; source parquet files were not overwritten. Two changed records belong to local training and one to local development. No new VQA pairs, boxes, masks, temporal labels or pseudo-labels were created for this checkpoint.

This repository discloses the change mechanism, counts, source-file checksums and annotation schemas. It excludes the exact restricted record IDs and answer values, original QA tables, source videos, and per-video LapChole metadata. The current LapChole data-use agreement prohibits public release of data and underlying annotations. Publicly releasing these values would therefore require clarification or permission from the organizers.

Local bookkeeping includes a seed-2026 complete-video split with rare-subtype protection and deterministic sampling schedules. These introduce no new ground-truth supervision. Aggregate split sizes are 104 training videos / 22,400 QA records and 26 development videos / 5,600 QA records. Only FRAME and SEGMENT are used.

The code license does not license the source datasets or their annotations. Obtain those resources from the original providers and comply with their current access and use terms.
