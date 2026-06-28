"""Generate the thesis plate domains (port of TestCaseGenerator.m).

Domains are generated in THIS package's coordinate convention (i->Y rows, j->X cols),
which is the TRANSPOSE of the MATLAB InversionToolbox convention (i->X, j->Y). So the
200x100 mm plate is `(nI=Y=104, nJ=X=204)` here, and a MATLAB defect `data(X0:X1, Y0:Y1)`
(1-based, inclusive) becomes Python `data[Y0-1:Y1, X0-1:X1]` (rows=Y, cols=X).

Labels (both defects are low-velocity vs the 6420 m/s aluminum background):
0 = ghost, 1 = aluminum (6420 m/s), 2 = crack (4800 m/s, the slowest), 3 = slow
inclusion (5000 m/s, used by the logo).
Files are written in the 3-header-row + column-major (Fortran) format that
`fwi.domain.read_domain` consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GHOST = 2
OFFSET = (-2.0, -2.0, 0.0)
SPACING = (1.0, 1.0, 1.0)
PLATE = (104, 204)  # (nI=Y, nJ=X): 100mm tall x 200mm wide + ghost
SMALL = (24, 54)  # cheap grid for gradient/Taylor/inversion tests

# (filename stem -> builder) for generate_all
DOMAINS: dict[str, str] = {
    "homogeneous": "uniform aluminum plate (FWI start model)",
    "cracked": "single vertical crack",
    "two_cracks": "two cracks",
    "three_cracks": "three cracks",
    "logo": "multi-defect 'L i square' logo (2 materials)",
    "small_homogeneous": "small uniform plate (tests)",
    "small_defect": "small plate + one defect (tests)",
}


def _base(shape: tuple[int, int]) -> np.ndarray:
    """Aluminum (label 1) interior with a 2-cell ghost (label 0) border."""
    a = np.ones(shape, dtype=np.int64)
    a[:GHOST, :] = 0
    a[-GHOST:, :] = 0
    a[:, :GHOST] = 0
    a[:, -GHOST:] = 0
    return a


def homogeneous(shape: tuple[int, int] = PLATE) -> np.ndarray:
    return _base(shape)


def cracked(shape: tuple[int, int] = PLATE) -> np.ndarray:
    # MATLAB data(80:110, 49:51) [X,Y] -> here rows Y 49..51, cols X 80..110
    a = _base(shape)
    a[48:51, 79:110] = 2
    return a


def two_cracks(shape: tuple[int, int] = PLATE) -> np.ndarray:
    # MATLAB case 4: data(80:100,49:51); data(112:113,46:56)
    a = _base(shape)
    a[48:51, 79:100] = 2
    a[45:56, 111:113] = 2
    return a


def three_cracks(shape: tuple[int, int] = PLATE) -> np.ndarray:
    # MATLAB case 5: data(80:100,49:51); data(105:108,62:63); data(115:116,46:56)
    a = _base(shape)
    a[48:51, 79:100] = 2
    a[61:63, 104:108] = 2
    a[45:56, 114:116] = 2
    return a


def logo(shape: tuple[int, int] = PLATE) -> np.ndarray:
    """Recreated thesis 'L i square' logo, all low-velocity vs aluminum: L (label 3,
    5000 m/s), i bar+dot (label 2, 4800 m/s crack-speed), square (label 3, 5000 m/s)."""
    a = _base(shape)
    # L (label 3, 5000 m/s): vertical stroke + horizontal foot
    a[30:81, 20:25] = 3
    a[76:81, 20:56] = 3
    # i (label 2, 4800 m/s): vertical bar + dot above
    a[40:76, 90:95] = 2
    a[30:35, 90:95] = 2
    # square (label 3, 5000 m/s)
    a[40:76, 140:176] = 3
    return a


def small_homogeneous(shape: tuple[int, int] = SMALL) -> np.ndarray:
    return _base(shape)


def small_defect(shape: tuple[int, int] = SMALL) -> np.ndarray:
    a = _base(shape)
    a[10:13, 24:30] = 2
    return a


_BUILDERS = {
    "homogeneous": homogeneous,
    "cracked": cracked,
    "two_cracks": two_cracks,
    "three_cracks": three_cracks,
    "logo": logo,
    "small_homogeneous": small_homogeneous,
    "small_defect": small_defect,
}


def write_domain(path: str | Path, labels: np.ndarray) -> None:
    """Write labels (nI,nJ) in the 3-header-row + column-major format read_domain expects."""
    nI, nJ = labels.shape
    flat = labels.flatten(order="F")  # column-major, matches read_domain's order='F'
    lines = [
        f"{OFFSET[0]:f} {OFFSET[1]:f} {OFFSET[2]:f}",
        f"{SPACING[0]:f} {SPACING[1]:f} {SPACING[2]:f}",
        f"{nI} {nJ} 1",
        " ".join(str(int(v)) for v in flat),
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def generate_all(data_dir: str | Path) -> None:
    """Generate every domain `.txt` into `data_dir`."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, builder in _BUILDERS.items():
        write_domain(d / f"{name}.txt", builder())


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "domain"
    generate_all(out)
    print(f"generated {len(_BUILDERS)} domains in {out}")
