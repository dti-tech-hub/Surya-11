"""Communication efficiency: update compression + byte measurement.

Compression methods (config ``communication.method``):

- ``none``           — full float32 payload (standard FedAvg baseline)
- ``topk``           — keep top-k fraction by magnitude; send values + int32 indices
- ``quantize``       — symmetric int8 quantisation with per-tensor scale
- ``topk_quantize``  — top-k sparsification followed by int8 quantisation

Payload sizes are *measured* from the actual serialised bytes (values + indices
+ per-tensor metadata header), never hard-coded.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import torch

from utilities.config import get

INT_DTYPE = np.int32


def state_to_arrays(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}


def arrays_to_state(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in arrays.items()}


def _header_bytes(header: dict[str, Any]) -> bytes:
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def serialize_payload(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    hb = _header_bytes(header)
    buf.write(len(hb).to_bytes(4, "little"))
    buf.write(hb)
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        buf.write(arr.tobytes())
    return buf.getvalue()


def compress_state(state: dict[str, torch.Tensor], cfg: dict) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Compress a state dict; returns (header, payload arrays, stats)."""
    method = str(get(cfg, "communication.method", "topk_quantize"))
    topk_fraction = float(get(cfg, "communication.topk_fraction", 0.2))
    quant_bits = int(get(cfg, "communication.quant_bits", 8))
    if quant_bits != 8:
        raise ValueError("Only quant_bits=8 is implemented")
    arrays = state_to_arrays(state)

    if method == "none":
        header = {
            "method": "none",
            "tensors": {
                key: {"shape": list(arr.shape), "sparse": False, "quantized": False}
                for key, arr in arrays.items()
            },
        }
        # Keep the same payload key convention as compressed methods
        # (``<key>__values``); decompress_state uses it for every method.
        payload = {f"{key}__values": arr for key, arr in arrays.items()}
        return header, payload, {"kept_fraction": 1.0, "quantized": False, "method": "none"}

    header: dict[str, Any] = {"method": method, "tensors": {}}
    payload: dict[str, np.ndarray] = {}
    kept_total, size_total = 0, 0

    for key in sorted(arrays):
        flat = arrays[key].reshape(-1)
        size_total += flat.size
        if method in ("topk", "topk_quantize"):
            k = max(1, int(round(topk_fraction * flat.size)))
            k = min(k, flat.size)
            idx = np.argpartition(np.abs(flat), -k)[-k:]
            vals = flat[idx]
            order = np.argsort(-np.abs(vals))
            idx, vals = idx[order].astype(INT_DTYPE), vals[order].astype(np.float32)
        else:  # pure quantize
            idx = None
            vals = flat
        if method in ("quantize", "topk_quantize"):
            max_abs = float(np.max(np.abs(vals))) if vals.size else 0.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            q = np.clip(np.round(vals / scale), -127, 127).astype(np.int8)
            header["tensors"][key] = {
                "shape": list(arrays[key].shape),
                "scale": scale,
                "sparse": idx is not None,
                "quantized": True,
            }
            payload[f"{key}__values"] = q
            if idx is not None:
                payload[f"{key}__idx"] = idx
        else:
            header["tensors"][key] = {"shape": list(arrays[key].shape), "sparse": idx is not None, "quantized": False}
            payload[f"{key}__values"] = vals
            if idx is not None:
                payload[f"{key}__idx"] = idx
        kept_total += int(vals.size)

    stats = {
        "kept_fraction": kept_total / max(size_total, 1),
        "quantized": method in ("quantize", "topk_quantize"),
        "method": method,
    }
    return header, payload, stats


def decompress_state(header: dict[str, Any], payload: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Reconstruct a dense state dict from a compressed payload."""
    out: dict[str, torch.Tensor] = {}
    method = header.get("method", "none")
    for key, meta in header["tensors"].items():
        shape = tuple(meta["shape"])
        size = int(np.prod(shape)) if shape else 1
        vals = payload.get(f"{key}__values")
        if vals is None:
            vals = payload[key]
        if meta.get("quantized") or method in ("quantize", "topk_quantize"):
            vals = vals.astype(np.float32) * float(meta.get("scale", 1.0))
        else:
            vals = vals.astype(np.float32)
        idx = payload.get(f"{key}__idx")
        if idx is not None and meta.get("sparse"):
            flat = np.zeros(size, dtype=np.float32)
            flat[idx.astype(np.int64)] = vals
            out[key] = torch.from_numpy(flat.reshape(shape))
        else:
            out[key] = torch.from_numpy(vals.reshape(shape))
    return out


def payload_bytes(header: dict[str, Any], payload: dict[str, np.ndarray]) -> int:
    """Actual serialised payload size in bytes (header + arrays)."""
    return len(serialize_payload(header, payload))


def uncompressed_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


class CommunicationMeter:
    """Accumulates per-round and total communication statistics."""

    def __init__(self) -> None:
        self.rounds: list[dict[str, Any]] = []

    def record_round(
        self,
        round_idx: int,
        per_client_bytes_down: list[int],
        per_client_bytes_up: list[int],
        duration_s: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "round": round_idx,
            "clients": len(per_client_bytes_up),
            "bytes_download": int(sum(per_client_bytes_down)),
            "bytes_upload": int(sum(per_client_bytes_up)),
            "bytes_per_round": int(sum(per_client_bytes_down) + sum(per_client_bytes_up)),
            "avg_bytes_per_client": float(np.mean(per_client_bytes_up + per_client_bytes_down)) if per_client_bytes_up else 0.0,
            "round_duration_s": float(duration_s),
        }
        if extra:
            entry.update(extra)
        self.rounds.append(entry)
        return entry

    def summary(self, uncompressed_reference_bytes: int | None = None) -> dict[str, Any]:
        if not self.rounds:
            return {"rounds": 0, "total_bytes": 0}
        total = int(sum(r["bytes_per_round"] for r in self.rounds))
        result: dict[str, Any] = {
            "rounds": len(self.rounds),
            "total_bytes": total,
            "bytes_per_round_mean": float(total / len(self.rounds)),
            "bytes_per_round": [r["bytes_per_round"] for r in self.rounds],
            "avg_bytes_per_client": float(np.mean([r["avg_bytes_per_client"] for r in self.rounds])),
            "mean_round_duration_s": float(np.mean([r["round_duration_s"] for r in self.rounds])),
        }
        if uncompressed_reference_bytes:
            baseline = uncompressed_reference_bytes * len(self.rounds) * max((r["clients"] for r in self.rounds), default=0) * 2
            result["uncompressed_reference_bytes"] = int(baseline)
            result["communication_reduction_pct"] = (
                100.0 * (1.0 - total / baseline) if baseline else 0.0
            )
        return result
