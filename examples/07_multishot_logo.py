"""Example 7: more recordings -> better results (moving-source multi-shot).

A single source moving from sensor to sensor, with the recordings combined, reconstructs
the multi-defect "L i square" logo better than one source position alone - the thesis
multi-shot acquisition. Both runs use the SAME moving-source acquisition and the SAME
frequency continuation (50 -> 100 -> 200 kHz, each stage seeded by the previous, avoiding
cold-start cycle-skipping); the only difference is how many source positions are combined:

  - 1 position : one boundary sensor fires, the others record (limited illumination);
  - N positions: each of N sensors fires in turn, the others record, and the misfit and
    gradient sum over the shots (the medium is illuminated from many angles).

Run: uv run python examples/07_multishot_logo.py [--shots 8] [--iters 10]
     [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import plotting
from fwi.config import SimConfig, resolve_dtype
from fwi.inversion import invert_multishot
from fwi.problems import build_multishot_problem, build_problem

OUT = Path("outputs")
FREQS = [50_000.0, 100_000.0, 200_000.0]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().cpu().double()
    b = b.flatten().cpu().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _continuation(n_shots, *, device, dtype, start, n_iter):
    """Invert the logo with `n_shots` moving-source positions via 50->100->200 kHz."""
    alpha2 = start.clone()
    n_used = 0
    for f0 in FREQS:
        cfg = SimConfig(f0=f0)
        mp = build_multishot_problem(
            "logo", device=device, dtype=dtype, cfg=cfg, n_shots=n_shots
        )
        n_used = len(mp.shots)
        alpha2, _ = invert_multishot(
            alpha2, mp.shots, mp.src_sig, cfg, active_mask=mp.active_mask, n_iter=n_iter
        )
    return alpha2.detach(), n_used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)
    ref = build_problem("logo", device=device, dtype=dtype, cfg=SimConfig(f0=FREQS[-1]))
    true_update = ref.true_alpha2 - ref.start_alpha2
    print(
        f"logo, frequency continuation {[int(f / 1e3) for f in FREQS]} kHz | "
        f"1 source position vs {args.shots} | {args.iters} iters/stage"
    )

    one_hat, _ = _continuation(1, device=device, dtype=dtype,
                               start=ref.start_alpha2, n_iter=args.iters)
    corr_one = _corr(one_hat - ref.start_alpha2, true_update)
    print(f"1 position : corr(true) = {corr_one:.3f}")

    many_hat, n_used = _continuation(args.shots, device=device, dtype=dtype,
                                     start=ref.start_alpha2, n_iter=args.iters)
    corr_many = _corr(many_hat - ref.start_alpha2, true_update)
    print(f"{n_used} positions: corr(true) = {corr_many:.3f}")

    plotting.save_field((one_hat - ref.start_alpha2), OUT / "multishot_single_source.png",
                        title=f"1 source position (corr={corr_one:.2f})")
    plotting.save_field((many_hat - ref.start_alpha2), OUT / "multishot_moving_source.png",
                        title=f"{n_used} source positions (corr={corr_many:.2f})")
    plotting.save_field(true_update, OUT / "multishot_true.png", title="true logo")
    print(f"saved reconstructions to {OUT}/")
    print(f"more recordings improve correlation by {corr_many - corr_one:+.3f}")

    assert corr_many > corr_one, (
        f"{n_used} positions ({corr_many:.3f}) did not beat 1 position ({corr_one:.3f})"
    )


if __name__ == "__main__":
    main()
