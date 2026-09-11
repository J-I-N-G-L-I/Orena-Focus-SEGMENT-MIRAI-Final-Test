# Public reconstruction recipe. This is not the original submitted image.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/app/requirements.txt \
    && python -c "import torch, torchvision, transformers, peft, decord; assert torch.__version__ == '2.8.0+cu128'; assert torch.version.cuda == '12.8'"

COPY runtime /opt/app/raw_runtime
COPY model/adapter /opt/app/model/adapter
COPY model/processor /opt/app/model/processor
COPY model/run_manifest.json /opt/app/model/run_manifest.json
COPY resources/base_qwen35_9b /opt/app/resources/base_qwen35_9b
RUN chmod -R a+rX /opt/app && mkdir -p /input /output && chmod 777 /output

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/tmp/raw_hf \
    TRITON_CACHE_DIR=/tmp/raw_triton \
    XDG_CACHE_HOME=/tmp/raw_cache \
    PYTHONPATH=/opt/app/raw_runtime \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CRD_ATTENTION=sdpa \
    CRD_SUBMISSION_PROFILE=legacy-fallback \
    TOKENIZERS_PARALLELISM=false \
    RAW_CHECKPOINT=/opt/app/model \
    RAW_BASE_MODEL=/opt/app/resources/base_qwen35_9b
WORKDIR /opt/app/raw_runtime
USER 999:999
ENTRYPOINT ["python", "/opt/app/raw_runtime/submission/inference.py"]
CMD []
