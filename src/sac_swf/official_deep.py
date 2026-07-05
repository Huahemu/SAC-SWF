"""Adapters for author-released deep denoisers used in Experiment 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

from .utils import rgb_to_gray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CODE = PROJECT_ROOT / "external" / "official_code"
OFFICIAL_WEIGHTS = PROJECT_ROOT / "external" / "official_weights"


@dataclass
class OfficialDeepSpec:
    method: str
    display_name: str
    paper: str
    author_repo: str
    execution_repo: str
    code_paths: list[str]
    weight_path: str
    weight_url: str
    expected_min_bytes: int
    status: str = "unchecked"
    note: str = ""


SPECS = {
    "official_dncnn": OfficialDeepSpec(
        method="official_dncnn",
        display_name="DnCNN",
        paper="Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising",
        author_repo="https://github.com/cszn/DnCNN",
        execution_repo="https://github.com/cszn/KAIR",
        code_paths=[
            "external/official_code/cszn_KAIR/models/network_dncnn.py",
            "external/official_code/cszn_KAIR/models/basicblock.py",
            "external/official_code/cszn_KAIR/utils/utils_image.py",
        ],
        weight_path="external/official_weights/kair/dncnn_25.pth",
        weight_url="https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth",
        expected_min_bytes=2_000_000,
    ),
    "official_drunet": OfficialDeepSpec(
        method="official_drunet",
        display_name="DRUNet",
        paper="Plug-and-Play Image Restoration with Deep Denoiser Prior",
        author_repo="https://github.com/cszn/DPIR",
        execution_repo="https://github.com/cszn/DPIR",
        code_paths=[
            "external/official_code/cszn_DPIR/models/network_unet.py",
            "external/official_code/cszn_DPIR/models/basicblock.py",
            "external/official_code/cszn_DPIR/utils/utils_image.py",
        ],
        weight_path="external/official_weights/dpir/drunet_gray.pth",
        weight_url="https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth",
        expected_min_bytes=120_000_000,
    ),
    "official_restormer": OfficialDeepSpec(
        method="official_restormer",
        display_name="Restormer",
        paper="Restormer: Efficient Transformer for High-Resolution Image Restoration",
        author_repo="https://github.com/swz30/Restormer",
        execution_repo="https://github.com/swz30/Restormer",
        code_paths=[
            "external/official_code/swz30_Restormer/basicsr/models/archs/restormer_arch.py",
            "external/official_code/swz30_Restormer/demo.py",
            "external/official_code/swz30_Restormer/release_v1.0.json",
        ],
        weight_path="external/official_weights/restormer/gaussian_gray_denoising_sigma25.pth",
        weight_url="https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_sigma25.pth",
        expected_min_bytes=100_000_000,
    ),
}


def _abs(rel: str) -> Path:
    return PROJECT_ROOT / rel


def inspect_official_deep_specs() -> list[dict]:
    """Return code/weight readiness for all official deep methods."""
    rows = []
    for spec in SPECS.values():
        code_status = []
        for rel in spec.code_paths:
            path = _abs(rel)
            code_status.append({"path": rel, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
        weight = _abs(spec.weight_path)
        actual = weight.stat().st_size if weight.exists() else 0
        ready = all(item["exists"] and item["bytes"] > 0 for item in code_status) and actual >= spec.expected_min_bytes
        rows.append(
            {
                **asdict(spec),
                "code_ready": all(item["exists"] and item["bytes"] > 0 for item in code_status),
                "weight_ready": actual >= spec.expected_min_bytes,
                "weight_actual_bytes": actual,
                "ready_for_inference": ready,
                "code_status": code_status,
            }
        )
    return rows


def _reset_models_namespace() -> None:
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]


def _image_to_tensor_gray(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = rgb_to_gray(image).astype(np.float32)
    return torch.from_numpy(arr)[None, None].to(device)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().clamp(0.0, 1.0).numpy()
    return arr[0, 0]


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)


def _crop_to_shape(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return x[..., :h, :w]


_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_official_dncnn(image: np.ndarray) -> np.ndarray:
    """Run KAIR official DnCNN network definition with dncnn_25.pth."""
    spec = SPECS["official_dncnn"]
    weight = _abs(spec.weight_path)
    if weight.stat().st_size < spec.expected_min_bytes:
        raise RuntimeError(f"Missing official DnCNN weight: {weight}")
    device = _device()
    if "official_dncnn" not in _MODEL_CACHE:
        _reset_models_namespace()
        repo = str(OFFICIAL_CODE / "cszn_KAIR")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        module = importlib.import_module("models.network_dncnn")
        model = module.DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="R")
        state = torch.load(weight, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _MODEL_CACHE["official_dncnn"] = model
    x = _image_to_tensor_gray(image, device)
    with torch.no_grad():
        y = _MODEL_CACHE["official_dncnn"](x)
    return _tensor_to_image(y)


def run_official_drunet(image: np.ndarray, noise_level: float = 25.0 / 255.0) -> np.ndarray:
    """Run DPIR official DRUNet when the official weight is available."""
    spec = SPECS["official_drunet"]
    weight = _abs(spec.weight_path)
    actual = weight.stat().st_size if weight.exists() else 0
    if actual < spec.expected_min_bytes:
        raise RuntimeError(f"Missing/incomplete official DRUNet weight: {weight} ({actual} bytes)")
    device = _device()
    if "official_drunet" not in _MODEL_CACHE:
        _reset_models_namespace()
        repo = str(OFFICIAL_CODE / "cszn_DPIR")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        module = importlib.import_module("models.network_unet")
        model = module.UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")
        state = torch.load(weight, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _MODEL_CACHE["official_drunet"] = model
    x = _image_to_tensor_gray(image, device)
    sigma = torch.ones_like(x[:, :1]) * float(noise_level)
    x2 = torch.cat([x, sigma], dim=1)
    x2, hw = _pad_to_multiple(x2, 8)
    with torch.no_grad():
        y = _MODEL_CACHE["official_drunet"](x2)
    return _tensor_to_image(_crop_to_shape(y, hw))


def run_official_restormer(image: np.ndarray) -> np.ndarray:
    """Run Restormer official architecture when the official weight is available."""
    spec = SPECS["official_restormer"]
    weight = _abs(spec.weight_path)
    actual = weight.stat().st_size if weight.exists() else 0
    if actual < spec.expected_min_bytes:
        raise RuntimeError(f"Missing/incomplete official Restormer weight: {weight} ({actual} bytes)")
    device = _device()
    if "official_restormer" not in _MODEL_CACHE:
        arch = OFFICIAL_CODE / "swz30_Restormer" / "basicsr" / "models" / "archs" / "restormer_arch.py"
        module_name = "restormer_arch_official"
        spec_import = importlib.util.spec_from_file_location(module_name, arch)
        if spec_import is None or spec_import.loader is None:
            raise RuntimeError(f"Cannot load Restormer architecture: {arch}")
        module = importlib.util.module_from_spec(spec_import)
        sys.modules[module_name] = module
        spec_import.loader.exec_module(module)
        model = module.Restormer(inp_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree")
        state = torch.load(weight, map_location="cpu")
        if isinstance(state, dict) and "params" in state:
            state = state["params"]
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _MODEL_CACHE["official_restormer"] = model
    x = _image_to_tensor_gray(image, device)
    x, hw = _pad_to_multiple(x, 8)
    with torch.no_grad():
        y = _MODEL_CACHE["official_restormer"](x)
    return _tensor_to_image(_crop_to_shape(y, hw))


def official_deep_methods() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Return official deep method callables in stable order."""
    return {
        "official_dncnn": run_official_dncnn,
        "official_drunet": run_official_drunet,
        "official_restormer": run_official_restormer,
    }
