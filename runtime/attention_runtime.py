"""FA2/efficient SDPA with the validated lab FLA path; portable inference fallback."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any

import torch


def select_attention() -> str:
    requested = os.environ.get("CRD_ATTENTION", "auto")
    if requested not in {"auto", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Invalid CRD_ATTENTION={requested}")
    if requested != "sdpa":
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            if callable(flash_attn_func) and callable(flash_attn_varlen_func):
                return "flash_attention_2"
        except (ImportError, OSError, RuntimeError):
            if requested == "flash_attention_2":
                raise
    return "sdpa"


ATTN_IMPLEMENTATION = select_attention()
EXPECTED_KERNEL_VERSIONS = {
    "flash-linear-attention": "0.2.2",
    "causal-conv1d": "1.5.0.post8",
    "triton": "3.1.0",
}
EXPECTED_LINEAR_LAYERS = 24
EXPECTED_CORE_VERSIONS = {
    "transformers": "5.9.0",
    "peft": "0.19.1",
    "wandb": "0.28.1",
}


def assert_lab_software_contract() -> dict[str, Any]:
    """Reject an environment that differs from the validated lab build."""

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, found {platform.python_version()}")
    if torch.__version__ != "2.5.1+cu121":
        raise RuntimeError(f"Expected torch 2.5.1+cu121, found {torch.__version__}")
    if bool(torch._C._GLIBCXX_USE_CXX11_ABI):
        raise RuntimeError("The laboratory kernel contract requires CXX11_ABI=FALSE")
    versions = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_CORE_VERSIONS
    }
    mismatches = {
        package: {"actual": versions[package], "expected": expected}
        for package, expected in EXPECTED_CORE_VERSIONS.items()
        if versions[package] != expected
    }
    if mismatches:
        raise RuntimeError(f"Laboratory core package contract failed: {mismatches}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cxx11_abi": False,
        "packages": versions,
    }


def linear_attention_fast_path_status(*, strict_versions: bool = True) -> tuple[bool, str, dict[str, str]]:
    """Validate versions, Transformers feature flags, and required symbols."""

    versions: dict[str, str] = {}
    try:
        versions = {
            package: importlib.metadata.version(package)
            for package in EXPECTED_KERNEL_VERSIONS
        }
        mismatches = {
            package: {"actual": versions[package], "expected": expected}
            for package, expected in EXPECTED_KERNEL_VERSIONS.items()
            if versions[package] != expected
        }
        if strict_versions and mismatches:
            return False, f"kernel version contract mismatch: {mismatches}", versions

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        causal_available = bool(is_causal_conv1d_available())
        fla_available = bool(is_flash_linear_attention_available())
        if not causal_available or not fla_available:
            return (
                False,
                f"causal_conv1d={causal_available}, flash_linear_attention={fla_available}",
                versions,
            )

        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        symbols = (
            causal_conv1d_fn,
            causal_conv1d_update,
            FusedRMSNormGated,
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        if not bool(is_fast_path_available):
            return False, "Transformers Qwen3.5 reports is_fast_path_available=False", versions
        if not all(symbol is not None for symbol in symbols):
            return False, "one or more required kernel symbols resolved to None", versions
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", versions
    return True, "causal-conv1d + flash-linear-attention", versions


def configure_lab_attention(*, require_fast_path: bool = True) -> dict[str, Any]:
    """Return and, for formal runs, enforce the complete attention policy."""

    available, detail, versions = linear_attention_fast_path_status(strict_versions=require_fast_path)
    if require_fast_path and not available:
        raise RuntimeError(
            "Qwen3.5 linear-attention fast path is required but unavailable: "
            f"{detail}. Run scripts/install_training_acceleration.sh and rerun the canary."
        )
    if not available and not require_fast_path:
        # Submission images may lack compatible FLA extensions. Explicitly bind
        # the native Torch implementations before constructing the model.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                     "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
            setattr(modeling, name, None)
        modeling.is_fast_path_available = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # A math fallback can allocate quadratic attention matrices at native
    # resolution. Fail visibly in the capacity probe instead of silently using it.
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda_backends = {
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    return {
        "implementation": ATTN_IMPLEMENTATION,
        "external_flash_attention_2": ATTN_IMPLEMENTATION == "flash_attention_2",
        "pytorch_sdpa_backends": cuda_backends,
        "linear_attention_fast_path": available,
        "linear_attention_detail": detail,
        "kernel_versions": versions,
    }


def audit_qwen_linear_attention_fast_path(model: torch.nn.Module) -> dict[str, Any]:
    """Audit the loaded architecture and the global fast-path binding."""

    available, detail, versions = linear_attention_fast_path_status()
    linear_modules = [
        name
        for name, module in model.named_modules()
        if "gateddeltanet" in type(module).__name__.replace("_", "").lower()
    ]
    if not available:
        raise RuntimeError(f"Loaded Qwen3.5 model cannot use the linear fast path: {detail}")
    if len(linear_modules) != EXPECTED_LINEAR_LAYERS:
        raise RuntimeError(
            "Unexpected Qwen3.5 linear-attention inventory: "
            f"{len(linear_modules)} != {EXPECTED_LINEAR_LAYERS}"
        )
    return {
        "implementation": ATTN_IMPLEMENTATION,
        "linear_attention_fast_path": True,
        "linear_attention_layers": len(linear_modules),
        "linear_attention_module_names": linear_modules,
        "detail": detail,
        "kernel_versions": versions,
    }
