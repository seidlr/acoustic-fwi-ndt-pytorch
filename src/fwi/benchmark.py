"""Speed-investigation variants and a timing harness (notebook 05).

Collects the forward-solver variants behind ONE uniform call shape so they can be verified
and benchmarked side by side, and keeps the timing logic (device sync + median) in one
tested place instead of inline in the notebook. Each `forward_*` returns traces; the
sequential/batched pair returns `(S, R, nt)`.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from fwi.config import SimConfig
from fwi.forward import forward, forward_multishot
from fwi.forward_conv2d import forward_conv2d
from fwi.rust_solver import rust_available, rust_forward

ForwardFn = Callable[..., torch.Tensor]


def forward_naive(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg) -> torch.Tensor:
    """Reference solver: F.pad + per-coefficient stencil loop (fwi.forward.forward)."""
    with torch.no_grad():
        return forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg).traces


def forward_conv2d_traces(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg) -> torch.Tensor:
    """Fused conv2d Laplacian (autograd- and GPU-friendly)."""
    with torch.no_grad():
        return forward_conv2d(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg).traces


def forward_rust_traces(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg) -> torch.Tensor:
    """Native Rust forward solve (CPU/float64)."""
    with torch.no_grad():
        return rust_forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)


def forward_variants(*, include_rust: bool = True) -> dict[str, ForwardFn]:
    """Name -> single-source forward callable (identical signature, returns traces).

    Rust is included only when the native extension is built.
    """
    variants: dict[str, ForwardFn] = {
        "naive": forward_naive,
        "conv2d": forward_conv2d_traces,
    }
    if include_rust and rust_available():
        variants["rust"] = forward_rust_traces
    return variants


def forward_sequential(
    alpha2, src_sig_1d, src_i, src_j, rec_i, rec_j, cfg: SimConfig
) -> torch.Tensor:
    """Baseline for `forward_multishot`: S single-source solves stacked -> (S, R, nt)."""
    sig = src_sig_1d.reshape(1, -1)
    with torch.no_grad():
        return torch.stack(
            [
                forward(alpha2, sig, src_i[s : s + 1], src_j[s : s + 1], rec_i, rec_j, cfg).traces
                for s in range(int(src_i.shape[0]))
            ]
        )


def forward_batched(
    alpha2, src_sig_1d, src_i, src_j, rec_i, rec_j, cfg: SimConfig
) -> torch.Tensor:
    """All S shots as one (S, nI, nJ) solve (fwi.forward.forward_multishot) -> (S, R, nt)."""
    with torch.no_grad():
        return forward_multishot(alpha2, src_sig_1d, src_i, src_j, rec_i, rec_j, cfg)


def device_sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so timing is honest (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_ms(fn: Callable[[], object], device: torch.device, *, k: int = 5) -> float:
    """Median ms/call over k runs (one warmup), device-synced for honest timing."""
    fn()
    device_sync(device)
    times: list[float] = []
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        device_sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]
