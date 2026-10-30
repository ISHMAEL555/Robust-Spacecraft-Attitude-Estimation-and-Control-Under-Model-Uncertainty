"""
Per-run truth-inertia sampling from the model-aided filter's inertia prior.

The six independent inertia elements ``[Jxx, Jyy, Jzz, Jxy, Jxz, Jyz]`` are drawn
from ``N(J_nom, P0_J)``, where ``P0_J = sigma^2 I_6`` in the filter's own
dimensionless coordinates: each element is scaled by ``[J11, J22, J33,
sqrt(J11 J22), sqrt(J11 J33), sqrt(J22 J33)]`` of the nominal tensor
(``MEKF._initialize_inertia_scales``). With the true-minus-estimate error
convention of the filter, ``J_true = J_nom + sum_j delta_j s_j E_j`` with
``delta ~ N(0, sigma^2 I_6)``, so the filter's initial inertia error is a draw
from its own prior and the initial 15-state NEES is chi-square(15).

A sample must be a physically realisable inertia tensor: symmetric positive
definite, with principal moments obeying the triangle inequality
(``I1 + I2 >= I3`` and permutations). Non-realisable draws are rejected and
redrawn. Rejection truncates the Gaussian slightly; the rejection count is
returned so it can be reported.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Basis order matches MEKF: [Jxx, Jyy, Jzz, Jxy, Jxz, Jyz].
_ELEMENT_INDEX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def inertia_element_scales(nominal: np.ndarray) -> np.ndarray:
    """Scales of the six dimensionless inertia coordinates (as in the MEKF)."""
    d = np.diag(np.asarray(nominal, dtype=float))
    return np.array([d[0], d[1], d[2],
                     np.sqrt(d[0] * d[1]), np.sqrt(d[0] * d[2]), np.sqrt(d[1] * d[2])])


def inertia_from_relative_errors(nominal: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """``J_nom + sum_j relative_j s_j E_j`` for the six dimensionless errors."""
    nominal = np.asarray(nominal, dtype=float)
    scales = inertia_element_scales(nominal)
    inertia = nominal.copy()
    for value, scale, (i, j) in zip(relative, scales, _ELEMENT_INDEX):
        inertia[i, j] += value * scale
        if i != j:
            inertia[j, i] += value * scale
    return inertia


def is_physically_realisable(inertia: np.ndarray, tol: float = 0.0) -> bool:
    """Symmetric, positive-definite, with principal moments satisfying I1 + I2 >= I3."""
    inertia = np.asarray(inertia, dtype=float)
    if not np.allclose(inertia, inertia.T):
        return False
    moments = np.linalg.eigvalsh(inertia)  # ascending
    if moments[0] <= 0.0:
        return False
    return bool(moments[0] + moments[1] >= moments[2] - tol)


def sample_inertia_from_prior(
    nominal: np.ndarray,
    relative_sigma: float,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> Tuple[np.ndarray, int]:
    """
    Draw a realisable truth inertia from the filter prior.

    Returns ``(inertia, rejections)``: ``rejections`` is the number of
    non-realisable draws discarded before the accepted one.
    """
    if relative_sigma < 0.0:
        raise ValueError("relative_sigma must be non-negative")
    nominal = np.asarray(nominal, dtype=float)
    rejections = 0
    for _ in range(max_attempts):
        relative = rng.standard_normal(6) * relative_sigma
        candidate = inertia_from_relative_errors(nominal, relative)
        if is_physically_realisable(candidate):
            return candidate, rejections
        rejections += 1
    raise RuntimeError(
        f"no realisable inertia in {max_attempts} draws; relative_sigma="
        f"{relative_sigma} is too large for this nominal tensor"
    )
