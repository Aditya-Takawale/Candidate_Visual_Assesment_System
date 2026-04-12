"""
Hardware-Aware Backend Selector
Auto-detects the best available ONNX Runtime execution provider.
Priority: CUDA > DirectML > CoreML/MPS > CPU
No manual user input required.
"""

from __future__ import annotations
import platform
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── CUDA provider options — tuned for laptop RTX 3050 (4 GB VRAM) ─────────────
_CUDA_PROVIDER_OPTIONS = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,   # 2 GB cap — leave room for other tasks
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "do_copy_in_default_stream": True,
}


def _cuda_available_ort() -> bool:
    """Check if CUDA provider is actually functional in onnxruntime-gpu."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _cuda_available_torch() -> bool:
    """Secondary check via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _log_gpu_info() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory // (1024 * 1024)
            logger.info(
                f"GPU: {props.name} | VRAM: {total_mb} MB | "
                f"CUDA {torch.version.cuda} | SM {props.major}.{props.minor}"
            )
    except Exception:
        pass


def get_available_providers() -> List[str]:
    """Return ordered list of ONNX Runtime execution providers for this machine."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        logger.warning("onnxruntime not installed — falling back to CPU-only mode.")
        return ["CPUExecutionProvider"]

    ordered: List[str] = []
    system = platform.system()

    if "CUDAExecutionProvider" in available:
        ordered.append("CUDAExecutionProvider")
        _log_gpu_info()
        logger.info("Hardware backend: NVIDIA CUDA (onnxruntime-gpu).")

    elif system == "Darwin":
        if "CoreMLExecutionProvider" in available:
            ordered.append("CoreMLExecutionProvider")
            logger.info("Hardware backend: Apple CoreML (Metal) detected.")
        elif "MPSExecutionProvider" in available:
            ordered.append("MPSExecutionProvider")
            logger.info("Hardware backend: Apple MPS detected.")

    elif system == "Windows" and "DmlExecutionProvider" in available:
        ordered.append("DmlExecutionProvider")
        logger.info("Hardware backend: Windows DirectML detected.")

    ordered.append("CPUExecutionProvider")
    if len(ordered) == 1:
        logger.info("Hardware backend: CPU only (no GPU provider found).")

    return ordered


def get_provider_options() -> list:
    """
    Return (providers, provider_options) tuple ready for ort.InferenceSession.
    Injects CUDA options when CUDA is the primary provider.
    """
    providers = get_available_providers()
    options = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            options.append(_CUDA_PROVIDER_OPTIONS)
        else:
            options.append({})
    return providers, options


def get_primary_backend() -> str:
    """Return a human-readable label for the active backend."""
    providers = get_available_providers()
    mapping = {
        "CUDAExecutionProvider": "CUDA (RTX)",
        "CoreMLExecutionProvider": "CoreML (Metal)",
        "MPSExecutionProvider": "MPS (Metal)",
        "DmlExecutionProvider": "DirectML",
        "CPUExecutionProvider": "CPU",
    }
    return mapping.get(providers[0], providers[0])


def get_torch_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'. Used by YOLOv8/ultralytics."""
    return "cuda" if _cuda_available_torch() else "cpu"


def build_ort_session(model_path: str):
    """Build an ONNX Runtime InferenceSession using the best available provider."""
    import onnxruntime as ort
    providers, options = get_provider_options()
    logger.debug(f"Loading ONNX model '{model_path}' with providers: {providers}")
    return ort.InferenceSession(model_path, providers=providers, provider_options=options)
