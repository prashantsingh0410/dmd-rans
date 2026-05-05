"""
dmd.py — Dynamic Mode Decomposition (DMD)
==========================================
Performs exact DMD on a snapshot data matrix produced by dataProcessing.py.

Algorithm reference:
    Tu et al. (2014) "On Dynamic Mode Decomposition: Theory and Applications"
    Journal of Computational Dynamics, 1(2), 391–421.

Typical usage
-------------
    from dataProcessing import load_data_matrix

    data_matrix, delta_t = load_data_matrix("/path/to/openfoam")
    result = compute_dmd(data_matrix, delta_t)
    print(result.summary())

Or run as a standalone script (uses a synthetic data matrix for smoke-testing):

    python dmd.py --delta-t 0.001 --rank 50

Exit codes: 0 = success, 1 = input-validation error, 2 = numerical error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch as pt

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a human-readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DMDResult:
    """
    Container for all quantities produced by a DMD computation.

    Attributes
    ----------
    modes : pt.Tensor
        Complex DMD mode matrix of shape (n_space, n_modes).
    eigenvalues : pt.Tensor
        Discrete-time eigenvalues λ_i (complex), shape (n_modes,).
    omega : pt.Tensor
        Continuous-time eigenvalues ω_i = log(λ_i)/Δt (complex), shape (n_modes,).
    frequencies : pt.Tensor
        Physical frequencies f_i = Im(ω_i) / (2π) in Hz, shape (n_modes,).
    amplitudes : pt.Tensor
        Mode amplitudes |b_i| computed from the initial condition, shape (n_modes,).
    growth_rates : pt.Tensor
        Modal growth / decay rates Re(ω_i), shape (n_modes,).
    positive_freq_indices : pt.Tensor
        Indices of modes whose frequency is strictly positive (unique spectrum half).
    singular_values : pt.Tensor
        Singular values retained from the truncated SVD, shape (r,).
    rank : int
        Effective truncation rank r used in the SVD.
    delta_t : float
        Temporal spacing Δt between snapshots (seconds).
    """

    modes: pt.Tensor
    eigenvalues: pt.Tensor
    omega: pt.Tensor
    frequencies: pt.Tensor
    amplitudes: pt.Tensor
    growth_rates: pt.Tensor
    positive_freq_indices: pt.Tensor
    singular_values: pt.Tensor
    rank: int
    delta_t: float
    # Internal: reduced linear operator — kept for downstream diagnostics
    _reduced_operator: pt.Tensor = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        top_n = min(5, self.positive_freq_indices.numel())
        idx = self.positive_freq_indices[
            self.amplitudes[self.positive_freq_indices].topk(top_n).indices
        ]
        lines = [
            "=== DMD Result ===",
            f"  Rank (r)          : {self.rank}",
            f"  Δt                : {self.delta_t:.6g} s",
            f"  Modes shape       : {tuple(self.modes.shape)}",
            f"  Positive-freq modes: {self.positive_freq_indices.numel()}",
            f"  Top-{top_n} dominant frequencies (Hz):",
        ]
        for i in idx.tolist():
            lines.append(
                f"    mode {i:4d} | f = {self.frequencies[i].item():+.4f} Hz"
                f" | |b| = {self.amplitudes[i].item():.4e}"
                f" | growth = {self.growth_rates[i].item():+.4e} s⁻¹"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core DMD routine
# ---------------------------------------------------------------------------

def compute_dmd(
    data_matrix: pt.Tensor,
    delta_t: float,
    rank: Optional[int] = None,
    mean_center: bool = True,
    device: Optional[pt.device] = None,
) -> DMDResult:
    """
    Compute exact DMD on *data_matrix*.

    Parameters
    ----------
    data_matrix : pt.Tensor
        Snapshot matrix of shape (n_space, n_time).  Each column is one
        flow-field snapshot at a uniformly-spaced time instant.
    delta_t : float
        Temporal spacing Δt between consecutive snapshots (seconds).
        Must be strictly positive.
    rank : int, optional
        Truncation rank r for the SVD.  If *None* (default) the full
        economy SVD rank is used (min(n_space, n_time - 1)).
    mean_center : bool
        If *True* (default), subtract the column-mean from *data_matrix*
        before decomposition.  This is the recommended practice for most
        CFD datasets.
    device : pt.device, optional
        Torch device on which computations are performed.  Defaults to
        the device of *data_matrix*.

    Returns
    -------
    DMDResult
        Fully populated result container; see :class:`DMDResult`.

    Raises
    ------
    TypeError
        If *data_matrix* is not a ``pt.Tensor`` or *delta_t* is not a
        positive float/int.
    ValueError
        If *data_matrix* has fewer than 2 columns or *rank* is out of
        range.
    RuntimeError
        If a numerical failure occurs during SVD or eigendecomposition.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    logger.debug("Validating inputs …")

    if not isinstance(data_matrix, pt.Tensor):
        raise TypeError(
            f"data_matrix must be a torch.Tensor, got {type(data_matrix).__name__}."
        )
    if data_matrix.ndim != 2:
        raise ValueError(
            f"data_matrix must be 2-D (n_space × n_time), got shape {tuple(data_matrix.shape)}."
        )
    n_space, n_time = data_matrix.shape
    if n_time < 2:
        raise ValueError(
            f"data_matrix must have at least 2 columns (snapshots), got {n_time}."
        )

    if not isinstance(delta_t, (int, float)) or delta_t <= 0:
        raise TypeError(
            f"delta_t must be a strictly positive number, got {delta_t!r}."
        )
    delta_t = float(delta_t)

    max_rank = min(n_space, n_time - 1)
    if rank is not None:
        if not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be a positive integer, got {rank!r}.")
        if rank > max_rank:
            raise ValueError(
                f"rank ({rank}) exceeds max allowable rank ({max_rank}) for this matrix."
            )
    else:
        rank = max_rank  # use full economy rank

    logger.info(
        "Input OK — shape: (%d, %d), Δt=%.4g s, rank=%d, mean_center=%s",
        n_space, n_time, delta_t, rank, mean_center,
    )

    # ------------------------------------------------------------------
    # 2. Transfer to target device; work on a float copy
    # ------------------------------------------------------------------
    if device is None:
        device = data_matrix.device

    X_full: pt.Tensor = data_matrix.to(device=device, dtype=pt.float64).clone()

    # ------------------------------------------------------------------
    # 3. Mean-centering (optional)
    # ------------------------------------------------------------------
    if mean_center:
        logger.debug("Mean-centering data matrix …")
        X_full -= X_full.mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # 4. Build snapshot pair (X, X′)
    # ------------------------------------------------------------------
    # X  = [x_0, x_1, …, x_{N-1}]   shape (n_space, N)
    # X' = [x_1, x_2, …, x_N  ]     shape (n_space, N)
    # where N = n_time - 1
    X: pt.Tensor = X_full[:, :-1].clone()   # all but last snapshot
    Xp: pt.Tensor = X_full[:, 1:].clone()   # all but first snapshot
    logger.debug("Snapshot pairs: X %s, X' %s", tuple(X.shape), tuple(Xp.shape))

    # ------------------------------------------------------------------
    # 5. Truncated SVD of X: X ≈ U Σ V^T
    # ------------------------------------------------------------------
    logger.info("Computing truncated SVD (rank=%d) …", rank)
    try:
        U_full, sigma_full, Vt_full = pt.linalg.svd(X, full_matrices=False)
    except Exception as exc:
        raise RuntimeError(f"SVD failed: {exc}") from exc

    # Truncate to desired rank
    U: pt.Tensor = U_full[:, :rank]        # (n_space, r)
    sigma: pt.Tensor = sigma_full[:rank]   # (r,)
    Vt: pt.Tensor = Vt_full[:rank, :]      # (r, N)
    V: pt.Tensor = Vt.T                    # (N, r)

    logger.debug(
        "Singular value range after truncation: [%.4e, %.4e]",
        sigma[-1].item(), sigma[0].item(),
    )

    sigma_inv: pt.Tensor = pt.diag(1.0 / sigma)   # (r, r)

    # ------------------------------------------------------------------
    # 6. Reduced linear operator  Ã = U^T X' V Σ⁻¹
    #    Ã approximates the dynamics in the r-dimensional POD subspace.
    # ------------------------------------------------------------------
    logger.info("Building reduced linear operator …")
    A_tilde: pt.Tensor = U.T @ Xp @ V @ sigma_inv   # (r, r)

    # ------------------------------------------------------------------
    # 7. Eigendecomposition of Ã: Ã W = W Λ
    # ------------------------------------------------------------------
    logger.info("Computing eigendecomposition of reduced operator …")
    try:
        eigenvalues: pt.Tensor
        W: pt.Tensor
        eigenvalues, W = pt.linalg.eig(A_tilde)   # W: (r, r), eigenvalues: (r,)
    except Exception as exc:
        raise RuntimeError(f"Eigendecomposition failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 8. Reconstruct full-space (exact) DMD modes
    #    Φ = X' V Σ⁻¹ W
    # ------------------------------------------------------------------
    logger.info("Reconstructing full-space DMD modes …")
    Phi: pt.Tensor = (Xp @ V @ sigma_inv).to(pt.cdouble) @ W   # (n_space, r)

    # ------------------------------------------------------------------
    # 9. Continuous-time eigenvalues and frequencies
    #    ω_i = log(λ_i) / Δt
    # ------------------------------------------------------------------
    omega: pt.Tensor = pt.log(eigenvalues) / delta_t          # (r,) complex
    frequencies: pt.Tensor = omega.imag / (2.0 * np.pi)       # Hz, real
    growth_rates: pt.Tensor = omega.real                       # s⁻¹, real

    positive_freq_indices: pt.Tensor = (frequencies > 0).nonzero().flatten()
    logger.info(
        "Positive-frequency modes: %d / %d", positive_freq_indices.numel(), rank
    )

    # ------------------------------------------------------------------
    # 10. Mode amplitudes from initial condition
    #     b = Φ† x_0
    # ------------------------------------------------------------------
    logger.info("Computing mode amplitudes from initial condition …")
    x0: pt.Tensor = X[:, 0].to(Phi.dtype)
    b: pt.Tensor = pt.linalg.pinv(Phi) @ x0   # (r,) complex
    amplitudes: pt.Tensor = b.abs()            # real, (r,)

    logger.info("DMD computation complete.")

    return DMDResult(
        modes=Phi,
        eigenvalues=eigenvalues,
        omega=omega,
        frequencies=frequencies,
        amplitudes=amplitudes,
        growth_rates=growth_rates,
        positive_freq_indices=positive_freq_indices,
        singular_values=sigma,
        rank=rank,
        delta_t=delta_t,
        _reduced_operator=A_tilde,
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def dominant_modes(result: DMDResult, n: int = 10) -> pt.Tensor:
    """
    Return the indices of the *n* most energetic positive-frequency modes,
    ranked by decreasing amplitude.

    Parameters
    ----------
    result : DMDResult
    n : int
        Number of modes to return.

    Returns
    -------
    pt.Tensor
        1-D tensor of mode indices (into ``result.modes``), length ≤ n.
    """
    pos_idx = result.positive_freq_indices
    if pos_idx.numel() == 0:
        logger.warning("No positive-frequency modes found.")
        return pt.tensor([], dtype=pt.long)

    n = min(n, pos_idx.numel())
    top_local = result.amplitudes[pos_idx].topk(n).indices
    return pos_idx[top_local]


def reconstruct_field(
    result: DMDResult,
    t: float,
    mode_indices: Optional[pt.Tensor] = None,
) -> pt.Tensor:
    """
    Reconstruct the spatial field at time *t* using a subset of DMD modes.

    The reconstruction formula is:
        x(t) = Σ_i  b_i · Φ_i · exp(ω_i · t)

    Parameters
    ----------
    result : DMDResult
    t : float
        Time instant (seconds) at which to evaluate the reconstruction.
    mode_indices : pt.Tensor, optional
        1-D integer tensor of mode indices to include.  If *None*, all
        modes are used.

    Returns
    -------
    pt.Tensor
        Reconstructed field, shape (n_space,), complex dtype.
        Take `.real` for physical fields.
    """
    idx = mode_indices if mode_indices is not None else pt.arange(result.rank)

    Phi_sub = result.modes[:, idx]             # (n_space, k)
    omega_sub = result.omega[idx]              # (k,)
    b_sub = (pt.linalg.pinv(result.modes) @ result.modes[:, 0])[idx]  # approximate

    # Time dynamics: exp(ω_i · t)
    dynamics = pt.exp(omega_sub * t)           # (k,) complex

    reconstructed: pt.Tensor = Phi_sub @ (b_sub * dynamics)
    return reconstructed


# ---------------------------------------------------------------------------
# CLI entry point (smoke-test / standalone usage)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic Mode Decomposition — standalone smoke-test.\n"
            "Generates a synthetic data matrix and runs DMD on it.\n"
            "For production use, import compute_dmd() from this module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.001,
        metavar="DT",
        help="Temporal spacing between snapshots in seconds (default: 0.001).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        metavar="R",
        help="Truncation rank for the SVD.  Defaults to full economy rank.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        metavar="N",
        help="Number of spatial degrees of freedom in the synthetic matrix (default: 500).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=200,
        metavar="M",
        help="Number of time snapshots in the synthetic matrix (default: 200).",
    )
    parser.add_argument(
        "--no-mean-center",
        action="store_true",
        help="Disable mean-centering of the data matrix.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging output.",
    )
    return parser


def _synthetic_data_matrix(
    n_rows: int,
    n_cols: int,
    delta_t: float,
    seed: int = 42,
) -> pt.Tensor:
    """
    Generate a synthetic low-rank snapshot matrix composed of two
    travelling-wave modes plus noise, useful for unit-testing.

    Parameters
    ----------
    n_rows, n_cols : int
        Spatial and temporal dimensions.
    delta_t : float
        Temporal spacing used to set the wave periods.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pt.Tensor  shape (n_rows, n_cols), float64
    """
    rng = pt.Generator()
    rng.manual_seed(seed)

    t = pt.arange(n_cols, dtype=pt.float64) * delta_t
    x = pt.linspace(0.0, 2.0 * np.pi, n_rows, dtype=pt.float64)

    # Mode 1: f = 10 Hz, mild spatial variation
    f1 = 10.0
    mode1 = pt.sin(x).unsqueeze(1) * pt.cos(2.0 * np.pi * f1 * t).unsqueeze(0)

    # Mode 2: f = 25 Hz, faster spatial variation
    f2 = 25.0
    mode2 = 0.5 * pt.cos(2.0 * x).unsqueeze(1) * pt.sin(2.0 * np.pi * f2 * t).unsqueeze(0)

    # Low-level broadband noise
    noise = 0.05 * pt.randn(n_rows, n_cols, dtype=pt.float64, generator=rng)

    return mode1 + mode2 + noise


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(logging.DEBUG if args.debug else logging.INFO)
    logger.info("=== DMD standalone smoke-test ===")

    # ---- Try to import from dataProcessing; fall back to synthetic data ----
    data_matrix: pt.Tensor
    delta_t: float

    try:
        # When dataProcessing.py is present on the Python path, use it.
        from dataProcessing import load_data_matrix  # type: ignore[import]

        logger.info("dataProcessing module found — loading real data …")
        data_matrix, delta_t = load_data_matrix()
        logger.info("Loaded data matrix: shape %s, Δt=%.4g s", tuple(data_matrix.shape), delta_t)
    except ImportError:
        logger.warning(
            "dataProcessing.py not found on PYTHONPATH. "
            "Falling back to a synthetic (%d × %d) test matrix.",
            args.rows,
            args.cols,
        )
        data_matrix = _synthetic_data_matrix(args.rows, args.cols, args.delta_t)
        delta_t = args.delta_t

    # ---- Run DMD ----
    try:
        result = compute_dmd(
            data_matrix,
            delta_t=delta_t,
            rank=args.rank,
            mean_center=not args.no_mean_center,
        )
    except (TypeError, ValueError) as exc:
        logger.error("Input validation error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Numerical error during DMD: %s", exc)
        sys.exit(2)

    print(result.summary())

    top_idx = dominant_modes(result, n=10)
    logger.info(
        "Top dominant mode indices: %s",
        top_idx.tolist(),
    )

    sys.exit(0)