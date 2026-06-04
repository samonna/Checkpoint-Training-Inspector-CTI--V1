from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "network",
    "weights",
    "module",
)
OPTIMIZER_KEYS = (
    "optimizer_state_dict",
    "optimizer",
    "optim_state_dict",
    "opt_state_dict",
)


@dataclass
class LoadedCheckpoint:
    name: str
    raw: Any
    state_dict: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None


def safe_torch_load(source: Any, *, mmap: bool = False) -> Any:
    """Load PyTorch checkpoints on CPU with a safer weights_only first attempt."""
    kwargs = {"map_location": "cpu"}
    if isinstance(source, (str, Path)) and mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(source, weights_only=True, **kwargs)
    except Exception:
        return torch.load(source, weights_only=False, **kwargs)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _tensor_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {str(k): v.detach().cpu() for k, v in obj.items() if torch.is_tensor(v)}


def extract_state_dict(obj: Any, *, strip_module_prefix: bool = True) -> Dict[str, torch.Tensor]:
    """Extract a tensor-only model state_dict from common PyTorch checkpoint layouts."""
    candidate = obj
    if hasattr(candidate, "state_dict") and callable(candidate.state_dict):
        candidate = candidate.state_dict()

    if isinstance(candidate, dict):
        for key in STATE_DICT_KEYS:
            if key in candidate:
                nested = candidate[key]
                if hasattr(nested, "state_dict") and callable(nested.state_dict):
                    nested = nested.state_dict()
                if isinstance(nested, dict):
                    tensors = _tensor_state_dict(nested)
                    if tensors:
                        return _strip_module_prefix(tensors) if strip_module_prefix else tensors
        tensors = _tensor_state_dict(candidate)
        if tensors:
            return _strip_module_prefix(tensors) if strip_module_prefix else tensors
    raise ValueError("Could not find tensor weights/state_dict in checkpoint.")


def extract_optimizer_state(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    for key in OPTIMIZER_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return None


def extract_metadata(obj: Any, name: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"name": name, "epoch_from_name": parse_epoch(name)}
    if isinstance(obj, dict):
        for key in ("epoch", "step", "global_step", "loss", "val_loss", "best_metric", "accuracy", "lr"):
            value = obj.get(key)
            if isinstance(value, (int, float, str, bool)) or value is None:
                metadata[key] = value
    return metadata


def load_checkpoint_bytes(data: bytes, name: str = "uploaded_checkpoint") -> LoadedCheckpoint:
    raw = safe_torch_load(io.BytesIO(data))
    state = extract_state_dict(raw)
    return LoadedCheckpoint(name=name, raw=raw, state_dict=state, metadata=extract_metadata(raw, name), optimizer_state=extract_optimizer_state(raw))


def load_checkpoint_path(path: str | Path, *, mmap: bool = True) -> LoadedCheckpoint:
    p = Path(path)
    raw = safe_torch_load(p, mmap=mmap)
    state = extract_state_dict(raw)
    return LoadedCheckpoint(name=p.name, raw=raw, state_dict=state, metadata=extract_metadata(raw, p.name), optimizer_state=extract_optimizer_state(raw))


def parse_epoch(name: str) -> Optional[int]:
    patterns = [r"epoch[_\- ]?(\d+)", r"ep[_\- ]?(\d+)", r"ckpt[_\- ]?(\d+)", r"(\d+)"]
    lower = name.lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def sort_checkpoints(items: Iterable[Tuple[str, bytes]]) -> List[Tuple[str, bytes]]:
    def key_fn(item: Tuple[str, bytes]):
        epoch = parse_epoch(item[0])
        return (epoch is None, epoch if epoch is not None else 10**12, item[0])
    return sorted(items, key=key_fn)


def read_zip_checkpoints(data: bytes, suffixes=(".pt", ".pth", ".bin")) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() in suffixes:
                out.append((Path(info.filename).name, zf.read(info.filename)))
    return sort_checkpoints(out)
