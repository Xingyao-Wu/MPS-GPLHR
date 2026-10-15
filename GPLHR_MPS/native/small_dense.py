from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def _as_real_if_safe(matrix: np.ndarray, tolerance: float = 1.0e-12):
    """Return a real view when imaginary entries are only numerical noise."""

    array = np.asarray(matrix)
    scale = max(float(np.linalg.norm(array)), 1.0)
    imaginary_norm = float(np.linalg.norm(np.imag(array)))
    if imaginary_norm <= tolerance * scale:
        return np.asarray(np.real(array), dtype=np.float64), True
    return np.asarray(array, dtype=np.complex128), False


@dataclass(frozen=True)
class MetricWhitening:
    transform: np.ndarray
    eigenvalues: np.ndarray
    retained: int
    condition_number: float


def metric_whitener(metric: np.ndarray, relative_cutoff: float) -> MetricWhitening:
    metric, _ = _as_real_if_safe(metric)
    metric = 0.5 * (metric + metric.conj().T)
    values, vectors = scipy.linalg.eigh(metric, check_finite=True)
    largest = max(float(values[-1]), 0.0)
    if largest == 0.0:
        raise FloatingPointError("zero Gram matrix")
    keep = values > relative_cutoff * largest
    if not np.any(keep):
        raise FloatingPointError("Gram rank guard removed every direction")
    retained_values = values[keep]
    transform = vectors[:, keep] / np.sqrt(retained_values)[None, :]
    return MetricWhitening(
        transform=transform,
        eigenvalues=values,
        retained=int(np.count_nonzero(keep)),
        condition_number=float(retained_values[-1] / retained_values[0]),
    )


def _selection_radius(values: np.ndarray, target: complex, count: int) -> float:
    distances = np.sort(np.abs(np.asarray(values) - complex(target)))
    if not 0 < count < distances.size:
        raise ValueError("selection radius requires a strict subset")
    return float(0.5 * (distances[count - 1] + distances[count]))


def _schur_blocks(triangular: np.ndarray, real_form: bool) -> list[tuple[int, int]]:
    """Return atomic 1x1/2x2 diagonal blocks of a Schur form."""

    size = triangular.shape[0]
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < size:
        block_size = 1
        if (
            real_form
            and index + 1 < size
            and triangular[index + 1, index] != 0.0
        ):
            block_size = 2
        blocks.append((index, block_size))
        index += block_size
    return blocks


def _select_schur_blocks(
    triangular: np.ndarray,
    blocks: list[tuple[int, int]],
    target: complex,
    count: int,
    *,
    real_only: bool = False,
) -> np.ndarray | None:
    """Select exactly ``count`` dimensions without splitting real 2x2 blocks."""

    
    choices: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for block_index, (start, block_size) in enumerate(blocks):
        if real_only and block_size != 1:
            continue
        values = scipy.linalg.eigvals(
            triangular[start : start + block_size, start : start + block_size],
            check_finite=False,
        )
        block_cost = float(
            np.sum(np.abs(values - complex(target)), dtype=np.float64)
        )
        updated = dict(choices)
        for retained, (cost, selected_blocks) in choices.items():
            new_retained = retained + block_size
            if new_retained > count:
                continue
            candidate = (cost + block_cost, selected_blocks + (block_index,))
            incumbent = updated.get(new_retained)
            if incumbent is None or candidate < incumbent:
                updated[new_retained] = candidate
        choices = updated
    if count not in choices:
        return None

    selection = np.zeros(triangular.shape[0], dtype=np.int32)
    for block_index in choices[count][1]:
        start, block_size = blocks[block_index]
        selection[start : start + block_size] = 1
    return selection


def _explicitly_ordered_schur(
    matrix: np.ndarray,
    target: complex,
    active: int,
    *,
    real_form: bool,
    require_real: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """Order a Schur form with an explicit LAPACK selection mask."""

    output = "real" if real_form else "complex"
    triangular, vectors = scipy.linalg.schur(
        matrix, output=output, check_finite=True
    )
    if active == matrix.shape[0]:
        if require_real:
            blocks = _schur_blocks(triangular, real_form)
            real_roots = sum(block_size == 1 for _, block_size in blocks)
            if real_roots != active:
                raise FloatingPointError(
                    f"projected real Schur form has only {real_roots} resolved "
                    f"real 1x1 Ritz blocks, but {active} are required; unresolved "
                    "2x2 conjugate blocks are rejected"
                )
            return (
                triangular,
                vectors,
                real_form,
                "full_real_schur_all_roots_resolved_1x1",
            )
        return triangular, vectors, real_form, "full_schur"

    blocks = _schur_blocks(triangular, real_form)
    selection = _select_schur_blocks(
        triangular, blocks, target, active, real_only=require_real
    )
    selection_mode = (
        "explicit_real_1x1_schur_blocks"
        if require_real
        else "explicit_real_schur_blocks"
    )
    if selection is None:
        if require_real:
            raise FloatingPointError(
                f"projected real Schur form has fewer than {active} real "
                "1x1 Ritz blocks; unresolved conjugate blocks are rejected"
            )
        # An odd-dimensional real invariant subspace need not exist when the
        # spectrum consists only of conjugate pairs.  Complex Schur form has
        # scalar blocks and is the mathematically correct exact-count fallback.
        triangular, vectors = scipy.linalg.schur(
            np.asarray(matrix, dtype=np.complex128),
            output="complex",
            check_finite=True,
        )
        real_form = False
        blocks = _schur_blocks(triangular, real_form)
        selection = _select_schur_blocks(triangular, blocks, target, active)
        selection_mode = "explicit_complex_fallback"
    if selection is None:
        raise FloatingPointError(
            f"cannot form a {active}-dimensional Schur invariant subspace"
        )

    trsen = scipy.linalg.lapack.get_lapack_funcs("trsen", (triangular,))
    if real_form:
        (
            triangular,
            vectors,
            _,
            _,
            selected,
            _,
            _,
            info,
        ) = trsen(selection, triangular, vectors, job="N", wantq=1)
    else:
        triangular, vectors, _, selected, _, _, info = trsen(
            selection, triangular, vectors, job="N", wantq=1
        )
    if info != 0:
        raise FloatingPointError(
            f"LAPACK trsen Schur reordering failed with info={info}"
        )
    if selected != active:
        raise FloatingPointError(
            f"explicit Schur reordering selected {selected}, expected {active}"
        )
    return triangular, vectors, real_form, selection_mode


def _select_generalized_schur_blocks(
    aa: np.ndarray,
    bb: np.ndarray,
    blocks: list[tuple[int, int]],
    count: int,
    *,
    required_real_roots: int = 0,
) -> np.ndarray | None:
    """Select the largest generalized-eigenvalue scores block-atomically."""

    choices: dict[
        tuple[int, int], tuple[tuple[int, float], tuple[int, ...]]
    ] = {(0, 0): ((0, 0.0), ())}
    for block_index, (start, block_size) in enumerate(blocks):
        alpha, beta = scipy.linalg.eigvals(
            aa[start : start + block_size, start : start + block_size],
            bb[start : start + block_size, start : start + block_size],
            homogeneous_eigvals=True,
            check_finite=False,
        )
        scores = np.divide(
            np.abs(alpha),
            np.abs(beta),
            out=np.full(alpha.shape, np.inf, dtype=np.float64),
            where=np.abs(beta) > 0.0,
        )
        block_quality = (
            int(np.count_nonzero(np.isinf(scores))),
            float(np.sum(scores[np.isfinite(scores)], dtype=np.float64)),
        )
        updated = dict(choices)
        real_dimensions = 1 if block_size == 1 else 0
        for (retained, real_roots), (quality, selected_blocks) in choices.items():
            new_retained = retained + block_size
            if new_retained > count:
                continue
            new_real_roots = min(
                required_real_roots, real_roots + real_dimensions
            )
            candidate_quality = (
                quality[0] + block_quality[0],
                quality[1] + block_quality[1],
            )
            candidate_blocks = selected_blocks + (block_index,)
            key = (new_retained, new_real_roots)
            incumbent = updated.get(key)
            if (
                incumbent is None
                or candidate_quality > incumbent[0]
                or (
                    candidate_quality == incumbent[0]
                    and candidate_blocks < incumbent[1]
                )
            ):
                updated[key] = (candidate_quality, candidate_blocks)
        choices = updated
    key = (count, required_real_roots)
    if key not in choices:
        return None

    selection = np.zeros(aa.shape[0], dtype=np.int32)
    for block_index in choices[key][1]:
        start, block_size = blocks[block_index]
        selection[start : start + block_size] = 1
    return selection


def _explicitly_ordered_qz(
    projected_n: np.ndarray,
    projected_m: np.ndarray,
    retain: int,
    *,
    real_form: bool,
    required_real_roots: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """Order generalized Schur blocks with an explicit LAPACK mask."""

    output = "real" if real_form else "complex"
    aa, bb, left, right = scipy.linalg.qz(
        projected_n, projected_m, output=output, check_finite=True
    )
    if retain == projected_n.shape[0]:
        if required_real_roots:
            blocks = _schur_blocks(aa, real_form)
            real_roots = sum(block_size == 1 for _, block_size in blocks)
            if real_roots < required_real_roots:
                raise FloatingPointError(
                    f"full harmonic real QZ form contains only {real_roots} "
                    f"resolved real 1x1 roots, but {required_real_roots} are "
                    "required; unresolved 2x2 conjugate blocks are rejected"
                )
        alpha, beta = scipy.linalg.eigvals(
            aa, bb, homogeneous_eigvals=True, check_finite=False
        )
        selection_mode = (
            f"full_real_qz_with_{required_real_roots}_resolved_1x1_roots"
            if required_real_roots
            else "full_qz"
        )
        return aa, bb, alpha, beta, left, right, real_form, selection_mode

    blocks = _schur_blocks(aa, real_form)
    selection = _select_generalized_schur_blocks(
        aa,
        bb,
        blocks,
        retain,
        required_real_roots=required_real_roots,
    )
    selection_mode = (
        f"explicit_real_qz_blocks_with_{required_real_roots}_real_1x1_roots"
        if required_real_roots
        else "explicit_real_qz_blocks"
    )
    if selection is None:
        if required_real_roots:
            raise FloatingPointError(
                f"harmonic real QZ form cannot retain {retain} dimensions "
                f"containing {required_real_roots} real 1x1 roots"
            )
        aa, bb, left, right = scipy.linalg.qz(
            np.asarray(projected_n, dtype=np.complex128),
            np.asarray(projected_m, dtype=np.complex128),
            output="complex",
            check_finite=True,
        )
        real_form = False
        blocks = _schur_blocks(aa, real_form)
        selection = _select_generalized_schur_blocks(aa, bb, blocks, retain)
        selection_mode = "explicit_complex_qz_fallback"
    if selection is None:
        raise FloatingPointError(
            f"cannot form a {retain}-dimensional generalized Schur subspace"
        )

    tgsen = scipy.linalg.lapack.get_lapack_funcs("tgsen", (aa, bb))
    if real_form:
        (
            aa,
            bb,
            alpha_real,
            alpha_imaginary,
            beta,
            left,
            right,
            selected,
            _,
            _,
            _,
            info,
        ) = tgsen(selection, aa, bb, left, right, ijob=0, wantq=1, wantz=1)
        alpha = alpha_real + 1.0j * alpha_imaginary
    else:
        (
            aa,
            bb,
            alpha,
            beta,
            left,
            right,
            selected,
            _,
            _,
            _,
            info,
        ) = tgsen(selection, aa, bb, left, right, ijob=0, wantq=1, wantz=1)
    if info != 0:
        raise FloatingPointError(
            f"LAPACK tgsen QZ reordering failed with info={info}"
        )
    if selected != retain:
        raise FloatingPointError(
            f"explicit QZ reordering selected {selected}, expected {retain}"
        )
    return aa, bb, alpha, beta, left, right, real_form, selection_mode


def ordered_schur(
    matrix: np.ndarray,
    target: complex,
    active: int,
    *,
    require_real: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move the ``active`` eigenvalues nearest target to the leading block."""

    matrix, matrix_is_real = _as_real_if_safe(matrix)
    real_form = matrix_is_real and abs(complex(target).imag) <= 1.0e-14
    if require_real and not real_form:
        raise FloatingPointError(
            "real-Ritz mode requires a numerically real projected matrix and target"
        )
    size = matrix.shape[0]
    if matrix.shape != (size, size) or not 1 <= active <= size:
        raise ValueError("invalid Schur dimensions")
    triangular, vectors, real_form, selection_mode = _explicitly_ordered_schur(
        matrix,
        target,
        active,
        real_form=real_form,
        require_real=require_real,
    )
    if real_form and active < size:
        boundary_scale = max(float(np.linalg.norm(triangular)), 1.0)
        if abs(triangular[active, active - 1]) > 1.0e-12 * boundary_scale:
            raise FloatingPointError(
                "requested Schur split bisects a real 2x2 conjugate-pair block"
            )
    selected_values = scipy.linalg.eigvals(
        triangular[:active, :active], check_finite=False
    )
    selected_values = selected_values[
        np.argsort(np.abs(selected_values - complex(target)))
    ]
    if real_form:
        
        lower_leak = np.tril(triangular, -2)
    else:
        lower_leak = np.tril(triangular, -1)
    return vectors, triangular, {
        "dimension": size,
        "active_dimension": active,
        "schur_form": "real_quasi_triangular" if real_form else "complex_triangular",
        "selection_mode": selection_mode,
        "real_1x1_ritz_constraint": bool(require_real),
        "selected_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in selected_values
        ],
        "unitarity_error": float(
            np.linalg.norm(vectors.conj().T @ vectors - np.eye(size))
        ),
        "lower_triangle_norm": float(np.linalg.norm(lower_leak)),
    }


def ordered_harmonic_qz(
    projected_n: np.ndarray,
    projected_m: np.ndarray,
    retain: int,
    target: complex,
    *,
    required_real_roots: int = 0,
    return_srr_factors: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict
]:
    """Solve ``N y = nu M y`` and order the largest ``|nu|`` first."""

    projected_n, n_is_real = _as_real_if_safe(projected_n)
    projected_m, m_is_real = _as_real_if_safe(projected_m)
    real_form = (
        n_is_real
        and m_is_real
        and abs(complex(target).imag) <= 1.0e-14
    )
    if required_real_roots and not real_form:
        raise FloatingPointError(
            "real-root harmonic mode requires numerically real QZ matrices and target"
        )
    size = projected_n.shape[0]
    if projected_n.shape != (size, size) or projected_m.shape != (size, size):
        raise ValueError("harmonic QZ matrices must be square and conforming")
    if not 1 <= retain <= size:
        raise ValueError("invalid QZ retention count")

    (
        aa,
        bb,
        alpha,
        beta,
        left,
        right,
        real_form,
        selection_mode,
    ) = _explicitly_ordered_qz(
        projected_n,
        projected_m,
        retain,
        real_form=real_form,
        required_real_roots=required_real_roots,
    )
    nu = np.divide(
        alpha,
        beta,
        out=np.full(alpha.shape, np.inf + 0.0j, dtype=np.complex128),
        where=np.abs(beta) > 0.0,
    )
    theta = np.asarray(
        [
            complex(target) if np.isinf(value) else complex(target) + 1.0 / value
            for value in nu
        ],
        dtype=np.complex128,
    )
    info = {
        "dimension": size,
        "retained": retain,
        "qz_form": "real_quasi_triangular" if real_form else "complex_triangular",
        "selection_mode": selection_mode,
        "required_real_1x1_roots": int(required_real_roots),
        "selected_theta": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in theta[:retain]
        ],
        "right_unitarity_error": float(
            np.linalg.norm(right.conj().T @ right - np.eye(size))
        ),
        "left_unitarity_error": float(
            np.linalg.norm(left.conj().T @ left - np.eye(size))
        ),
    }
    if not return_srr_factors:
        return right[:, :retain], theta[:retain], info

    
    selected_right = right[:, :retain]
    selected_n = aa[:retain, :retain]
    selected_m = bb[:retain, :retain]
    if real_form and np.any(np.diag(selected_n, k=-1) != 0.0):
        
        selected_n, selected_m, _, local_right = scipy.linalg.qz(
            np.asarray(selected_n, dtype=np.complex128),
            np.asarray(selected_m, dtype=np.complex128),
            output="complex",
            check_finite=False,
        )
        selected_right = selected_right @ local_right
        info["selected_factor_form"] = "complex_triangular_from_real_2x2_block"
    else:
        info["selected_factor_form"] = "triangular_resolved_real_blocks"
    r_a = selected_m + complex(target) * selected_n
    r_b = selected_n
    diagonal_a = np.asarray(np.diag(r_a), dtype=np.complex128)
    diagonal_b = np.asarray(np.diag(r_b), dtype=np.complex128)
    use_ratio = np.abs(diagonal_a) < np.abs(diagonal_b)
    g1_diagonal = np.where(
        use_ratio,
        0.0,
        np.divide(
            1.0 - diagonal_b,
            diagonal_a,
            out=np.zeros_like(diagonal_a, dtype=np.complex128),
            where=np.abs(diagonal_a) > 0.0,
        ),
    )
    g2_diagonal = np.where(
        use_ratio,
        np.divide(
            1.0,
            diagonal_b,
            out=np.zeros_like(diagonal_b, dtype=np.complex128),
            where=np.abs(diagonal_b) > 0.0,
        ),
        1.0,
    )
    g1 = np.diag(g1_diagonal)
    g2 = np.diag(g2_diagonal)
    g = r_a @ g1 + r_b @ g2
    g_inverse_r_a = scipy.linalg.solve(
        g, r_a, assume_a="gen", check_finite=False
    )
    factor_a = g2 @ g_inverse_r_a
    factor_b = np.eye(retain, dtype=factor_a.dtype) - g1 @ g_inverse_r_a
    factor_values = scipy.linalg.eigvals(
        factor_a, factor_b, check_finite=False
    )
    info["srr_factor_eigenvalues"] = [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in factor_values
    ]
    info["srr_factor_relation"] = "A V M_B - V M_A"
    info["srr_factor_source"] = (
        "harmonic QZ factors via GPLHR Eqs. (2.8), (2.11), and (2.12)"
    )
    return selected_right, theta[:retain], factor_a, factor_b, info
