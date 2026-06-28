"""Plot helpers (headless Agg backend). Figures are saved, never shown.

Grows across tasks: wavefield/kernel/traces here; hockey-stick, convergence, and
reconstruction added by later tasks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUTPUTS = Path("outputs")


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_field(
    field, path, title: str = "", cmap: str = "seismic", symmetric: bool = True
):
    """Save a 2D field image (i->Y down, j->X across)."""
    f = _np(field)
    p = _ensure(path)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if symmetric:
        m = float(np.abs(f).max()) or 1.0
        im = ax.imshow(f, cmap=cmap, aspect="equal", vmin=-m, vmax=m)
    else:
        im = ax.imshow(f, cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x index (j)")
    ax.set_ylabel("y index (i)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def save_hockey_stick(
    hs, r1, r2, path, title: str = "Taylor / hockey-stick test", note: str = ""
):
    """Log-log plot of first/second-order Taylor remainders vs step size h."""
    hs = list(map(float, hs))
    p = _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, r1, "o-", label="r1 = |J(m+h dm) - J(m)|  (O(h))")
    ax.loglog(hs, r2, "s-", label="r2 = |... - h<g,dm>|  (O(h^2))")
    # reference slopes anchored at the largest h
    h0 = max(hs)
    i0 = hs.index(h0)
    if r1[i0] > 0:
        ax.loglog(hs, [r1[i0] * (h / h0) for h in hs], "k--", lw=0.8, label="slope 1")
    if r2[i0] > 0:
        ax.loglog(
            hs, [r2[i0] * (h / h0) ** 2 for h in hs], "k:", lw=0.8, label="slope 2"
        )
    ax.set_xlabel("step size h")
    ax.set_ylabel("Taylor remainder")
    ax.set_title(title)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def save_convergence(history, path, title: str = "inversion convergence"):
    """Semilog plot of misfit vs iteration."""
    h = _np(history)
    p = _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(h) + 1), h, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("waveform misfit J")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def save_traces(traces, path, title: str = "receiver traces"):
    """Save a (n_rec, nt) seismogram image."""
    t = _np(traces)
    p = _ensure(path)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    m = float(np.abs(t).max()) or 1.0
    im = ax.imshow(t, cmap="seismic", aspect="auto", vmin=-m, vmax=m)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("receiver")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def save_frequency_panel(
    models, freqs, path, title="inverted wave speed vs source frequency"
):
    """Thesis-style multi-panel: one recovered-speed image per source frequency."""
    p = _ensure(path)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2))
    if n == 1:
        axes = [axes]
    vmax = max(float(np.abs(_np(m)).max()) or 1.0 for m in models)
    for ax, m, f in zip(axes, models, freqs):
        im = ax.imshow(_np(m), cmap="seismic", aspect="equal", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{f / 1e3:.0f} kHz")
        ax.set_xlabel("x index (j)")
    axes[0].set_ylabel("y index (i)")
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(title)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
