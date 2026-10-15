from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from renormalizer import Mpo, Mps
from renormalizer.mps.block_env import BlockEnvData
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.lib import Environ, cvec2cmat
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.mps.svd_qn import add_outer, get_qn_mask
from renormalizer.utils import CompressConfig, CompressCriteria

from .block_one_site import block_hdiag_one_site, block_hop_expr_one_site
from .local_krylov import reliable_bicgstab
from .shifted_solver import (
    _absorb_coefficient,
    _norm,
    global_shifted_residual,
)


DIRECT_LOCAL_THRESHOLD = 256
logger = logging.getLogger(__name__)


@dataclass
class AMEnSolveReport:
    """Auditable diagnostics returned by :func:`solve_shifted_amen`."""

    shift: complex
    sweeps: int
    local_solves: int
    local_iterations: int
    local_failures: int
    max_active_dimension: int
    max_local_relative_residual: float
    global_relative_residual: float
    global_residual_history: list[float]
    converged: bool
    max_bond_dimension: int
    seconds: float
    use_block_env: bool = False
    block_local_solves: int = 0
    solver: str = "amen"
    local_matvecs: int = 0
    local_breakdowns: int = 0
    peak_bond_dimension: int = 0
    enrichment_events: int = 0
    stagnation_detected: bool = False
    half_sweeps: int = 0
    globally_certified: bool = True
    max_symmetry_block_dimension: int = 0
    symmetry_block_bond_cap: int | None = None


def _phase_for_shift(shift: complex) -> complex:
    if shift.imag > 0.0:
        return -1.0j
    if shift.imag < 0.0:
        return 1.0j
    return 1.0 + 0.0j


def _dense_environment(value):
    if isinstance(value, BlockEnvData):
        return value.to_dense()
    return np.asarray(asnumpy(value))


def _project_rhs_one_site(
    rhs: Mps,
    identity: Mpo,
    cross_environ: Environ,
    solution: Mps,
    site: int,
    left_index: int,
    right_index: int,
    left_method: str,
    right_method: str,
):
    """Project ``b`` into the current one-site canonical frame."""

    
    solution_conjugate = solution.conj()
    left = cross_environ.GetLR(
        "L",
        left_index,
        rhs,
        identity,
        method=left_method,
        mps_conj=solution_conjugate,
    )
    right = cross_environ.GetLR(
        "R",
        right_index,
        rhs,
        identity,
        method=right_method,
        mps_conj=solution_conjugate,
    )
    expression = hop_expr(
        left,
        right,
        [asxp(identity[site])],
        rhs[site].shape,
    )
    return expression(asxp(rhs[site])), left, right


def _candidate_columns(
    samples: np.ndarray,
    existing: np.ndarray,
    rank: int,
    relative_cutoff: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant sample-column space orthogonal to ``existing``.

    Exact thin SVD is used for small sectors.  Larger sectors use a
    deterministic oversampled randomized range followed by a small SVD.  The
    latter is important at L=20, where a residual interface can have thousands
    of columns but only two to four enrichment directions are required.
    """

    samples = np.asarray(samples, dtype=np.complex128)
    existing = np.asarray(existing, dtype=np.complex128)
    if samples.size == 0 or samples.shape[0] == 0 or samples.shape[1] == 0:
        return np.zeros((samples.shape[0], 0), dtype=samples.dtype), np.zeros(0)
    if existing.size:
        samples = samples - existing @ (existing.conj().T @ samples)
    sample_norm = float(np.linalg.norm(samples))
    if not np.isfinite(sample_norm) or sample_norm == 0.0:
        return np.zeros((samples.shape[0], 0), dtype=samples.dtype), np.zeros(0)

    requested = min(int(rank), min(samples.shape))
    if requested == 0:
        return np.zeros((samples.shape[0], 0), dtype=samples.dtype), np.zeros(0)
    if min(samples.shape) <= max(64, 4 * requested):
        vectors, values, _ = scipy.linalg.svd(
            samples, full_matrices=False, lapack_driver="gesdd"
        )
    else:
        range_rank = min(min(samples.shape), requested + 4)
        omega = rng.normal(size=(samples.shape[1], range_rank))
        omega = omega + 1.0j * rng.normal(
            size=(samples.shape[1], range_rank)
        )
        projected = samples @ omega
        if existing.size:
            projected -= existing @ (existing.conj().T @ projected)
        basis, _ = scipy.linalg.qr(projected, mode="economic")
        small = basis.conj().T @ samples
        small_left, values, _ = scipy.linalg.svd(
            small, full_matrices=False, lapack_driver="gesdd"
        )
        vectors = basis @ small_left

    if values.size == 0 or values[0] == 0.0:
        return vectors[:, :0], values[:0]
    keep = values > max(float(relative_cutoff), 1.0e-13) * values[0]
    count = min(requested, int(np.count_nonzero(keep)))
    return vectors[:, :count], values[:count]


def _select_sector_candidates(candidates, rank: int):
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[: int(rank)]


def _select_candidates_per_sector(candidates, rank: int):
    """Select up to ``rank`` enrichment directions in every QN block."""

    selected = []
    labels = sorted({item[1] for item in candidates})
    for label in labels:
        block = [item for item in candidates if item[1] == label]
        block.sort(key=lambda item: item[0], reverse=True)
        selected.extend(block[: int(rank)])
    return selected


def _maximum_qn_degeneracy(state: Mps) -> int:
    maximum = 1
    for labels in state.qn:
        _unique, counts = np.unique(
            np.asarray(labels), axis=0, return_counts=True
        )
        maximum = max(maximum, int(np.max(counts)))
    return maximum


def _enrich_lr(
    solution: Mps,
    rhs: Mps,
    hamiltonian: Mpo,
    identity: Mpo,
    site: int,
    operator_left,
    rhs_left,
    *,
    enrichment_rank: int,
    cutoff: float,
    rng: np.random.Generator,
    max_bond_per_qn: int | None = None,
) -> int:
    """Append QN-safe left residual directions at the cut after ``site``."""

    if site >= solution.site_num - 1:
        return 0
    if max_bond_per_qn is None:
        available = int(solution.bond_dims_exact[site + 1]) - int(
            solution.bond_dims[site + 1]
        )
        if available <= 0:
            return 0
        enrichment_rank = min(int(enrichment_rank), available)
    left_h = _dense_environment(operator_left)
    left_b = np.asarray(asnumpy(rhs_left))
    x_site = np.asarray(asnumpy(solution[site]))
    b_site = np.asarray(asnumpy(rhs[site]))
    h_site = np.asarray(asnumpy(hamiltonian[site]))
    i_site = np.asarray(asnumpy(identity[site]))

    h_interface = np.einsum(
        "abc,bdef,cek->adfk", left_h, h_site, x_site, optimize=True
    )
    b_interface = np.einsum(
        "abc,bdef,cek->adfk", left_b, i_site, b_site, optimize=True
    )
    rows = x_site.shape[0] * x_site.shape[1]
    samples = np.concatenate(
        [b_interface.reshape(rows, -1), h_interface.reshape(rows, -1)],
        axis=1,
    )
    current = x_site.reshape(rows, x_site.shape[-1])
    row_qn = add_outer(
        np.asarray(solution.qn[site]),
        np.asarray(solution._get_sigmaqn(site)),
    ).reshape(rows, -1)
    column_qn = np.asarray(solution.qn[site + 1])

    candidates = []
    for label in sorted({tuple(value) for value in row_qn.tolist()}):
        row_indices = np.flatnonzero(np.all(row_qn == label, axis=1))
        column_indices = np.flatnonzero(
            np.all(column_qn == label, axis=1)
        )
        existing = current[np.ix_(row_indices, column_indices)]
        requested = int(enrichment_rank)
        if max_bond_per_qn is not None:
            requested = min(
                requested,
                int(max_bond_per_qn) - int(column_indices.size),
            )
        if requested <= 0:
            continue
        vectors, values = _candidate_columns(
            samples[row_indices, :],
            existing,
            requested,
            cutoff,
            rng,
        )
        for vector, value in zip(vectors.T, values):
            candidates.append((float(value), label, row_indices, vector))

    if max_bond_per_qn is None:
        selected = _select_sector_candidates(candidates, enrichment_rank)
    else:
        selected = _select_candidates_per_sector(candidates, enrichment_rank)
    if not selected:
        return 0
    added = np.zeros((rows, len(selected)), dtype=np.complex128)
    labels = []
    for column, (_value, label, row_indices, vector) in enumerate(selected):
        added[row_indices, column] = vector
        labels.append(label)
    new_site = np.concatenate([current, added], axis=1).reshape(
        x_site.shape[0], x_site.shape[1], current.shape[1] + len(selected)
    )
    neighbor = np.asarray(asnumpy(solution[site + 1]))
    zeros = np.zeros(
        (len(selected), *neighbor.shape[1:]), dtype=np.result_type(neighbor, complex)
    )
    solution[site] = new_site
    solution[site + 1] = np.concatenate([neighbor, zeros], axis=0)
    solution.qn[site + 1] = np.concatenate(
        [column_qn, np.asarray(labels, dtype=column_qn.dtype)], axis=0
    )
    return len(selected)


def _enrich_rl(
    solution: Mps,
    rhs: Mps,
    hamiltonian: Mpo,
    identity: Mpo,
    site: int,
    operator_right,
    rhs_right,
    *,
    enrichment_rank: int,
    cutoff: float,
    rng: np.random.Generator,
    max_bond_per_qn: int | None = None,
) -> int:
    """Append QN-safe right residual directions at the cut before ``site``."""

    if site <= 0:
        return 0
    if max_bond_per_qn is None:
        available = int(solution.bond_dims_exact[site]) - int(
            solution.bond_dims[site]
        )
        if available <= 0:
            return 0
        enrichment_rank = min(int(enrichment_rank), available)
    right_h = _dense_environment(operator_right)
    right_b = np.asarray(asnumpy(rhs_right))
    x_site = np.asarray(asnumpy(solution[site]))
    b_site = np.asarray(asnumpy(rhs[site]))
    h_site = np.asarray(asnumpy(hamiltonian[site]))
    i_site = np.asarray(asnumpy(identity[site]))

    h_interface = np.einsum(
        "bdef,cek,lfk->bcdl", h_site, x_site, right_h, optimize=True
    )
    b_interface = np.einsum(
        "bdef,cek,lfk->bcdl", i_site, b_site, right_b, optimize=True
    )
    columns = x_site.shape[1] * x_site.shape[2]
    samples = np.concatenate(
        [b_interface.reshape(-1, columns), h_interface.reshape(-1, columns)],
        axis=0,
    )
    current = x_site.reshape(x_site.shape[0], columns)
    column_qn = add_outer(
        np.asarray(solution._get_sigmaqn(site)),
        np.asarray(solution.qn[site + 1]),
    ).reshape(columns, -1)
    row_qn = np.asarray(solution.qn[site])

    candidates = []
    for label in sorted({tuple(value) for value in column_qn.tolist()}):
        column_indices = np.flatnonzero(
            np.all(column_qn == label, axis=1)
        )
        row_indices = np.flatnonzero(np.all(row_qn == label, axis=1))
        existing_columns = current[np.ix_(row_indices, column_indices)].conj().T
        requested = int(enrichment_rank)
        if max_bond_per_qn is not None:
            requested = min(
                requested,
                int(max_bond_per_qn) - int(row_indices.size),
            )
        if requested <= 0:
            continue
        # Candidate columns are conjugate-transposed residual rows.
        vectors, values = _candidate_columns(
            samples[:, column_indices].conj().T,
            existing_columns,
            requested,
            cutoff,
            rng,
        )
        for vector, value in zip(vectors.T, values):
            candidates.append((float(value), label, column_indices, vector))

    if max_bond_per_qn is None:
        selected = _select_sector_candidates(candidates, enrichment_rank)
    else:
        selected = _select_candidates_per_sector(candidates, enrichment_rank)
    if not selected:
        return 0
    added = np.zeros((len(selected), columns), dtype=np.complex128)
    labels = []
    for row, (_value, label, column_indices, vector) in enumerate(selected):
        added[row, column_indices] = vector.conj()
        labels.append(label)
    new_site = np.concatenate([current, added], axis=0).reshape(
        current.shape[0] + len(selected), x_site.shape[1], x_site.shape[2]
    )
    neighbor = np.asarray(asnumpy(solution[site - 1]))
    zeros = np.zeros(
        (*neighbor.shape[:-1], len(selected)),
        dtype=np.result_type(neighbor, complex),
    )
    solution[site] = new_site
    solution[site - 1] = np.concatenate([neighbor, zeros], axis=-1)
    solution.qn[site] = np.concatenate(
        [row_qn, np.asarray(labels, dtype=row_qn.dtype)], axis=0
    )
    return len(selected)


def _effective_diagonal(left, right, local_mpo, qn_mask):
    left = _dense_environment(left)
    right = _dense_environment(right)
    local_mpo = np.asarray(asnumpy(local_mpo))
    left_diagonal = np.einsum("aba->ba", left)
    mpo_diagonal = np.einsum("abbc->abc", local_mpo)
    right_diagonal = np.einsum("aba->ba", right)
    diagonal = np.einsum(
        "ba,bcg,gf->acf",
        left_diagonal,
        mpo_diagonal,
        right_diagonal,
        optimize=True,
    )
    return np.asarray(diagonal)[qn_mask]


def _effective_matrix(left, right, local_mpo, qn_mask, shift, phase):
    """Materialize a small projected ``phase * (zI-H_eff)`` matrix."""

    left = _dense_environment(left)
    right = _dense_environment(right)
    local_mpo = np.asarray(asnumpy(local_mpo))
    tensor = np.einsum(
        "abc,bdef,lfk->adlcek",
        left,
        local_mpo,
        right,
        optimize=True,
    )
    full_dimension = int(np.prod(qn_mask.shape))
    indices = np.flatnonzero(np.asarray(qn_mask).reshape(-1))
    projected_h = tensor.reshape(full_dimension, full_dimension)[
        np.ix_(indices, indices)
    ]
    return phase * (
        complex(shift) * np.eye(indices.size, dtype=np.complex128) - projected_h
    )


def _half_sweep(
    solution: Mps,
    hamiltonian: Mpo,
    identity: Mpo,
    rhs: Mps,
    shift: complex,
    *,
    local_maxiter: int,
    local_tolerance: float,
    residual_replacement_interval: int,
    enrichment_rank: int,
    cutoff: float,
    use_block_env: bool,
    previous_global_residual: float | None,
    sweep_index: int,
    max_bond_per_qn: int | None,
    local_solver: str,
    gmres_initial_residual_tolerance: bool = False,
):
    """Perform one directional one-site solve/enrichment sweep."""

    environment_domain = "R" if solution.to_right else "L"
    operator_environ = Environ(
        solution,
        hamiltonian,
        environment_domain,
        use_block_env=use_block_env,
        block_env_min_bond_dim=0,
    )
    cross_environ = Environ(
        rhs,
        identity,
        environment_domain,
        mps_conj=solution.conj(),
    )
    phase = _phase_for_shift(shift)
    rng = np.random.default_rng(104729 * sweep_index + (0 if solution.to_right else 1))

    statistics = {
        "local_solves": 0,
        "local_iterations": 0,
        "local_failures": 0,
        "local_matvecs": 0,
        "local_breakdowns": 0,
        "block_local_solves": 0,
        "max_active_dimension": 0,
        "max_local_relative_residual": 0.0,
        "enrichment_events": 0,
        "peak_bond_dimension": max(int(value) for value in solution.bond_dims),
    }

    for site in solution.iter_idx_list(full=True):
        if solution.to_right:
            left_method, right_method = "System", "Enviro"
        else:
            left_method, right_method = "Enviro", "System"
        left_index = site - 1
        right_index = site + 1
        qn_big_left, qn_big_right, qn_matrix = solution._get_big_qn([site])
        qn_mask = np.asarray(get_qn_mask(qn_matrix, solution.qntot))
        active_dimension = int(np.count_nonzero(qn_mask))
        if active_dimension == 0:
            raise RuntimeError(f"empty local quantum-number sector at site {site}")
        statistics["max_active_dimension"] = max(
            statistics["max_active_dimension"], active_dimension
        )

        left = operator_environ.GetLR(
            "L",
            left_index,
            solution,
            hamiltonian,
            method=left_method,
            raw=use_block_env,
        )
        right = operator_environ.GetLR(
            "R",
            right_index,
            solution,
            hamiltonian,
            method=right_method,
            raw=use_block_env,
        )
        local_shape = tuple(qn_mask.shape)
        block_action = isinstance(left, BlockEnvData) and isinstance(
            right, BlockEnvData
        )
        if block_action:
            block_cache = getattr(operator_environ, "_block_env_cache", None)
            symbolic_keys = (
                None
                if block_cache is None
                else block_cache.symbolic_nonzero_keys(site)
            )
            expression = block_hop_expr_one_site(
                left,
                right,
                hamiltonian[site],
                local_shape,
                qn_mask,
                solution.qn[site],
                solution.qn[site + 1],
                hamiltonian.qn[site],
                hamiltonian.qn[site + 1],
                operator_environ.block_env_threshold,
                symbolic_keys,
            )
            if expression.size != active_dimension:
                raise RuntimeError("packed one-site layout has the wrong size")

            def apply(vector):
                return phase * expression.apply_shifted_packed(vector, shift)

            h_diagonal = block_hdiag_one_site(
                left,
                right,
                hamiltonian[site],
                local_shape,
                hamiltonian.qn[site],
                hamiltonian.qn[site + 1],
                operator_environ.block_env_threshold,
                symbolic_keys,
            )[qn_mask]
            statistics["block_local_solves"] += 1
        else:
            dense_left = _dense_environment(left)
            dense_right = _dense_environment(right)
            expression = hop_expr(
                asxp(dense_left),
                asxp(dense_right),
                [asxp(hamiltonian[site])],
                local_shape,
            )

            def apply(vector):
                structured = asxp(cvec2cmat(np.asarray(vector), qn_mask))
                image = np.asarray(asnumpy(expression(structured)))[qn_mask]
                return phase * (complex(shift) * np.asarray(vector) - image)

            h_diagonal = _effective_diagonal(
                dense_left, dense_right, hamiltonian[site], qn_mask
            )

        projected_rhs, rhs_left, rhs_right = _project_rhs_one_site(
            rhs,
            identity,
            cross_environ,
            solution,
            site,
            left_index,
            right_index,
            left_method,
            right_method,
        )
        projected_rhs = phase * np.asarray(asnumpy(projected_rhs))[qn_mask]
        initial = np.asarray(asnumpy(solution[site]))[qn_mask]
        correction_rhs = projected_rhs - apply(initial)
        statistics["local_matvecs"] += 1
        rhs_norm = float(np.linalg.norm(projected_rhs))
        initial_relative = float(
            np.linalg.norm(correction_rhs) / max(rhs_norm, 1.0e-300)
        )
        if previous_global_residual is None:
            adaptive_tolerance = max(local_tolerance, 1.0e-2)
        else:
            adaptive_tolerance = max(
                local_tolerance, min(1.0e-2, 0.1 * previous_global_residual)
            )

        shifted_diagonal = phase * (complex(shift) - h_diagonal)
        diagonal_scale = max(float(np.max(np.abs(shifted_diagonal))), 1.0)
        floor = 1.0e-10 * diagonal_scale
        safe_diagonal = np.asarray(shifted_diagonal, dtype=np.complex128).copy()
        small = np.abs(safe_diagonal) < floor
        safe_diagonal[small] = floor

        # Dense LU is both faster and more reliable for genuinely small
        # microproblems.  Large production centers stay matrix-free and use
        # constant-storage BiCGSTAB.
        full_local_dimension = int(np.prod(qn_mask.shape))
        use_dense_contraction = (
            active_dimension <= DIRECT_LOCAL_THRESHOLD
            and full_local_dimension <= 4 * active_dimension
        )
        use_column_direct = active_dimension <= 64
        if local_solver != "gmres" and (
            use_dense_contraction or use_column_direct
        ):
            if use_dense_contraction:
                local_matrix = _effective_matrix(
                    left,
                    right,
                    hamiltonian[site],
                    qn_mask,
                    shift,
                    phase,
                )
                assembly_matvecs = 0
            else:
                local_matrix = np.column_stack(
                    [
                        apply(np.eye(active_dimension, dtype=np.complex128)[:, j])
                        for j in range(active_dimension)
                    ]
                )
                assembly_matvecs = active_dimension
            try:
                delta = scipy.linalg.solve(
                    local_matrix,
                    correction_rhs,
                    assume_a="gen",
                    check_finite=False,
                )
                solved = initial + delta
                final_relative = float(
                    np.linalg.norm(projected_rhs - local_matrix @ solved)
                    / max(rhs_norm, 1.0e-300)
                )
                direct_failure = not np.isfinite(final_relative)
            except (scipy.linalg.LinAlgError, ValueError):
                solved = initial
                final_relative = initial_relative
                direct_failure = True
            local_iterations = 1
            local_matvecs = 1 + assembly_matvecs
            local_breakdowns = 0
            hard_failure = direct_failure
        elif local_solver == "bicgstab":
            correction = reliable_bicgstab(
                apply,
                correction_rhs,
                right_preconditioner=lambda vector: vector / safe_diagonal,
                rtol=0.0,
                atol=adaptive_tolerance * max(rhs_norm, 1.0e-300),
                maxiter=local_maxiter,
                residual_replacement_interval=residual_replacement_interval,
            )
            solved = initial + correction.solution
            final_relative = float(
                correction.relative_residual
                * np.linalg.norm(correction_rhs)
                / max(rhs_norm, 1.0e-300)
            )
            hard_failure = (
                (not np.isfinite(final_relative))
                or (
                    not correction.converged
                    and initial_relative > adaptive_tolerance
                    and final_relative >= 0.95 * initial_relative
                )
            )
            local_iterations = correction.iterations
            local_matvecs = correction.matvecs
            local_breakdowns = correction.breakdowns
            if hard_failure:
                fallback_matvecs = 0

                def counted_apply(vector):
                    nonlocal fallback_matvecs
                    fallback_matvecs += 1
                    return apply(vector)

                operator = scipy.sparse.linalg.LinearOperator(
                    (active_dimension, active_dimension),
                    matvec=counted_apply,
                    dtype=np.complex128,
                )
                preconditioner = scipy.sparse.linalg.LinearOperator(
                    (active_dimension, active_dimension),
                    matvec=lambda vector: vector / safe_diagonal,
                    dtype=np.complex128,
                )
                fallback_history: list[float] = []
                restart = min(32, active_dimension)
                fallback_delta, info = scipy.sparse.linalg.gmres(
                    operator,
                    correction_rhs,
                    x0=correction.solution,
                    M=preconditioner,
                    rtol=0.0,
                    atol=adaptive_tolerance * max(rhs_norm, 1.0e-300),
                    restart=restart,
                    maxiter=max(1, int(np.ceil(local_maxiter / restart))),
                    callback=lambda value: fallback_history.append(float(value)),
                    callback_type="pr_norm",
                )
                fallback_solution = initial + fallback_delta
                fallback_relative = float(
                    np.linalg.norm(projected_rhs - apply(fallback_solution))
                    / max(rhs_norm, 1.0e-300)
                )
                fallback_matvecs += 1
                if np.isfinite(fallback_relative) and (
                    info == 0 or fallback_relative < final_relative
                ):
                    solved = fallback_solution
                    final_relative = fallback_relative
                hard_failure = (
                    not np.isfinite(final_relative)
                    or final_relative >= 0.95 * initial_relative
                )
                local_iterations += len(fallback_history)
                local_matvecs += fallback_matvecs
        else:
            gmres_matvecs = 0

            def counted_apply(vector):
                nonlocal gmres_matvecs
                gmres_matvecs += 1
                return apply(vector)

            operator = scipy.sparse.linalg.LinearOperator(
                (active_dimension, active_dimension),
                matvec=counted_apply,
                dtype=np.complex128,
            )
            preconditioner = scipy.sparse.linalg.LinearOperator(
                (active_dimension, active_dimension),
                matvec=lambda vector: vector / safe_diagonal,
                dtype=np.complex128,
            )
            gmres_history: list[float] = []
            correction, info = scipy.sparse.linalg.gmres(
                operator,
                correction_rhs,
                M=preconditioner,
                rtol=0.0,
                atol=(local_tolerance * np.linalg.norm(correction_rhs)
                      if gmres_initial_residual_tolerance else
                      adaptive_tolerance * max(rhs_norm, 1.0e-300)),
                restart=min(int(local_maxiter), active_dimension),
                maxiter=1,
                callback=lambda value: gmres_history.append(float(value)),
                callback_type="pr_norm",
            )
            solved = initial + correction
            final_relative = float(
                np.linalg.norm(projected_rhs - apply(solved))
                / max(rhs_norm, 1.0e-300)
            )
            gmres_matvecs += 1
            hard_failure = (
                not np.isfinite(final_relative)
                or (
                    initial_relative > adaptive_tolerance
                    and final_relative >= 0.95 * initial_relative
                )
            )
            local_iterations = len(gmres_history)
            local_matvecs = gmres_matvecs
            local_breakdowns = int(info < 0)
        statistics["local_solves"] += 1
        statistics["local_iterations"] += local_iterations
        statistics["local_matvecs"] += local_matvecs
        statistics["local_breakdowns"] += local_breakdowns
        statistics["local_failures"] += int(hard_failure)
        statistics["max_local_relative_residual"] = max(
            statistics["max_local_relative_residual"], final_relative
        )

        structured_solution = asxp(cvec2cmat(solved, qn_mask))
        direction_to_right = bool(solution.to_right)
        solution._update_mps(
            structured_solution,
            [site],
            qn_big_left,
            qn_big_right,
            percent=0.0,
        )
        if local_solver == "gmres":
            added = 0
        elif direction_to_right:
            added = _enrich_lr(
                solution,
                rhs,
                hamiltonian,
                identity,
                site,
                left,
                rhs_left,
                enrichment_rank=enrichment_rank,
                cutoff=cutoff,
                rng=rng,
                max_bond_per_qn=max_bond_per_qn,
            )
        else:
            added = _enrich_rl(
                solution,
                rhs,
                hamiltonian,
                identity,
                site,
                right,
                rhs_right,
                enrichment_rank=enrichment_rank,
                cutoff=cutoff,
                rng=rng,
                max_bond_per_qn=max_bond_per_qn,
            )
        if added:
            statistics["enrichment_events"] += 1
            statistics["peak_bond_dimension"] = max(
                statistics["peak_bond_dimension"],
                max(int(value) for value in solution.bond_dims),
            )

    solution._switch_direction()
    return statistics


def solve_shifted_amen(
    hamiltonian: Mpo,
    rhs: Mps,
    shift: complex,
    x0: Mps | None = None,
    *,
    max_sweeps: int = 8,
    local_maxiter: int = 64,
    local_tolerance: float = 1.0e-8,
    bond_cap: int = 64,
    cutoff: float = 1.0e-12,
    global_tolerance: float = 1.0e-8,
    enrichment_rank: int = 4,
    residual_replacement_interval: int = 8,
    stagnation_sweeps: int = 2,
    use_block_env: bool = False,
    known_initial_residual: float | None = None,
    half_sweeps_per_sweep: int = 2,
    certify_global_residual: bool = True,
    symmetry_block_bond_cap: int | None = None,
    local_solver: str = "bicgstab",
    residual_evaluator=None,
    gmres_initial_residual_tolerance: bool = False,
) -> tuple[Mps, AMEnSolveReport]:
    """Solve ``(z I-H)x=b`` by direct one-site AMEn sweeps.

    By default ``max_sweeps`` counts complete left-to-right/right-to-left
    sweep pairs.  ``half_sweeps_per_sweep=1`` and
    ``certify_global_residual=False`` provide a deliberately cheap inexact
    application for an outer eigensolver preconditioner.  In that mode the
    caller must use an outer residual as the authoritative acceptance test.
    Local tolerances are relaxed early and tighten with the certified global
    residual.  ``local_failures`` counts only breakdown/no-reduction events;
    deliberately inexact but reducing micro-solves are accepted by AMEn.

    For GMRES, ``gmres_initial_residual_tolerance=True`` instead measures
    the requested local reduction against the initial local residual, as in
    Baiardi et al., arXiv:2110.00092, Eq. (22). ``residual_evaluator`` can
    replace the default global certificate without changing the equation.
    """

    operator_dims = tuple(int(value) for value in hamiltonian.pbond_list)
    rhs_dims = tuple(int(value) for value in rhs.pbond_list)
    if hamiltonian.site_num != rhs.site_num or operator_dims != rhs_dims:
        raise ValueError("Hamiltonian and right-hand side use incompatible models")
    if x0 is not None and (
        x0.site_num != rhs.site_num
        or tuple(int(value) for value in x0.pbond_list) != rhs_dims
    ):
        raise ValueError("initial guess and right-hand side use incompatible models")
    if min(
        max_sweeps,
        local_maxiter,
        bond_cap,
        residual_replacement_interval,
        stagnation_sweeps,
    ) < 1 or enrichment_rank < 0:
        raise ValueError(
            "sweep, bond, and iteration counts must be positive and "
            "enrichment_rank must be nonnegative"
        )
    if local_solver not in ("bicgstab", "gmres"):
        raise ValueError("local_solver must be 'bicgstab' or 'gmres'")
    if half_sweeps_per_sweep not in (1, 2):
        raise ValueError("half_sweeps_per_sweep must be one or two")
    if symmetry_block_bond_cap is not None and symmetry_block_bond_cap < 1:
        raise ValueError("symmetry_block_bond_cap must be positive")
    if not (0.0 < cutoff < 1.0):
        raise ValueError("cutoff must lie strictly between zero and one")
    if local_tolerance <= 0.0 or global_tolerance <= 0.0:
        raise ValueError("residual tolerances must be positive")
    if known_initial_residual is not None and (
        not np.isfinite(known_initial_residual) or known_initial_residual < 0.0
    ):
        raise ValueError("known_initial_residual must be finite and nonnegative")

    started = time.perf_counter()
    rhs_work = _absorb_coefficient(rhs)
    if _norm(rhs_work) == 0.0:
        raise ValueError("the shifted-system right-hand side is zero")
    if x0 is None:
        solution = rhs_work.copy()
        if abs(shift) > 1.0e-14:
            solution.scale(1.0 / complex(shift), inplace=True)
    else:
        solution = _absorb_coefficient(x0)
    solution.to_complex(inplace=True)
    aggregate_bond_cap = int(bond_cap)
    if symmetry_block_bond_cap is not None:
        aggregate_bond_cap = int(max(solution.bond_dims_exact))
    solution.compress_config = CompressConfig(
        (
            CompressCriteria.fixed
            if symmetry_block_bond_cap is not None
            else CompressCriteria.both
        ),
        threshold=float(cutoff),
        max_bonddim=aggregate_bond_cap,
        max_bonddim_per_qn=symmetry_block_bond_cap,
    )
    if symmetry_block_bond_cap is not None:
        if _maximum_qn_degeneracy(solution) > symmetry_block_bond_cap:
            raise ValueError(
                "initial AMEn state exceeds the symmetry-block bond cap"
            )
    elif max(solution.bond_dims) > bond_cap:
        solution.canonicalise().compress()
    solution.ensure_right_canonical()
    identity = Mpo.identity(hamiltonian.model)

    totals = {
        "local_solves": 0,
        "local_iterations": 0,
        "local_failures": 0,
        "local_matvecs": 0,
        "local_breakdowns": 0,
        "block_local_solves": 0,
        "max_active_dimension": 0,
        "max_local_relative_residual": 0.0,
        "enrichment_events": 0,
        "peak_bond_dimension": max(int(value) for value in solution.bond_dims),
    }
    residual_history: list[float] = []
    best_solution = (
        None if known_initial_residual is None else solution.copy()
    )
    best_residual = (
        float("inf")
        if known_initial_residual is None
        else float(known_initial_residual)
    )
    previous_residual = (
        None
        if known_initial_residual is None
        else float(known_initial_residual)
    )
    stagnant_sweep_count = 0
    stagnation_detected = False
    completed_sweeps = 0
    completed_half_sweeps = 0

    for sweep in range(1, max_sweeps + 1):
        for _direction in range(int(half_sweeps_per_sweep)):
            half = _half_sweep(
                solution,
                hamiltonian,
                identity,
                rhs_work,
                complex(shift),
                local_maxiter=int(local_maxiter),
                local_tolerance=float(local_tolerance),
                residual_replacement_interval=int(residual_replacement_interval),
                enrichment_rank=int(enrichment_rank),
                cutoff=float(cutoff),
                use_block_env=bool(use_block_env),
                previous_global_residual=previous_residual,
                sweep_index=2 * sweep + _direction,
                max_bond_per_qn=symmetry_block_bond_cap,
                local_solver=local_solver,
                gmres_initial_residual_tolerance=gmres_initial_residual_tolerance,
            )
            for key in (
                "local_solves",
                "local_iterations",
                "local_failures",
                "local_matvecs",
                "local_breakdowns",
                "block_local_solves",
                "enrichment_events",
            ):
                totals[key] += half[key]
            for key in (
                "max_active_dimension",
                "max_local_relative_residual",
                "peak_bond_dimension",
            ):
                totals[key] = max(totals[key], half[key])
            completed_half_sweeps += 1

        
        solution.ensure_right_canonical()
        if symmetry_block_bond_cap is None:
            solution.compress()
        solution.ensure_right_canonical()
        maximum_bond = max(int(value) for value in solution.bond_dims)
        maximum_block_bond = _maximum_qn_degeneracy(solution)
        if (
            symmetry_block_bond_cap is not None
            and maximum_block_bond > symmetry_block_bond_cap
        ):
            raise RuntimeError(
                "AMEn retained symmetry block exceeds its per-block cap"
            )
        if symmetry_block_bond_cap is None and maximum_bond > bond_cap:
            raise RuntimeError(
                f"AMEn retained bond {maximum_bond} exceeds cap {bond_cap}"
            )
        if not certify_global_residual:
            
            best_solution = solution.copy()
            completed_sweeps = sweep
            continue
        residual = (residual_evaluator or global_shifted_residual)(
            hamiltonian, rhs_work, solution, shift
        )
        residual_history.append(residual)
        completed_sweeps = sweep
        previous_residual = residual
        if residual < best_residual:
            materially_better = residual < (1.0 - 1.0e-3) * best_residual
            best_residual = residual
            best_solution = solution.copy()
            stagnant_sweep_count = 0 if materially_better else stagnant_sweep_count + 1
        else:
            stagnant_sweep_count += 1
        if stagnant_sweep_count >= stagnation_sweeps:
            stagnation_detected = True
        if residual <= global_tolerance:
            logger.info(
                "AMEn shift=%s sweep=%d residual=%.6e max_bond=%d converged",
                shift,
                sweep,
                residual,
                maximum_bond,
            )
            break
        logger.info(
            "AMEn shift=%s sweep=%d residual=%.6e best=%.6e max_bond=%d",
            shift,
            sweep,
            residual,
            best_residual,
            maximum_bond,
        )

    if best_solution is None:
        raise RuntimeError("AMEn completed no usable sweep")
    solution = best_solution
    if not certify_global_residual:
        best_residual = float("nan")
    solution.ensure_right_canonical()
    maximum_bond = max(int(value) for value in solution.bond_dims)
    report = AMEnSolveReport(
        shift=complex(shift),
        sweeps=completed_sweeps,
        local_solves=totals["local_solves"],
        local_iterations=totals["local_iterations"],
        local_failures=totals["local_failures"],
        max_active_dimension=totals["max_active_dimension"],
        max_local_relative_residual=totals["max_local_relative_residual"],
        global_relative_residual=best_residual,
        global_residual_history=residual_history,
        converged=(
            bool(certify_global_residual)
            and best_residual <= global_tolerance
        ),
        max_bond_dimension=maximum_bond,
        seconds=time.perf_counter() - started,
        use_block_env=bool(use_block_env),
        block_local_solves=totals["block_local_solves"],
        solver=("one-site-gmres" if local_solver == "gmres" else "amen"),
        local_matvecs=totals["local_matvecs"],
        local_breakdowns=totals["local_breakdowns"],
        peak_bond_dimension=totals["peak_bond_dimension"],
        enrichment_events=totals["enrichment_events"],
        stagnation_detected=stagnation_detected,
        half_sweeps=completed_half_sweeps,
        globally_certified=bool(certify_global_residual),
        max_symmetry_block_dimension=_maximum_qn_degeneracy(solution),
        symmetry_block_bond_cap=symmetry_block_bond_cap,
    )
    return solution, report


def solve_shifted_one_site_gmres(
    hamiltonian: Mpo,
    rhs: Mps,
    shift: complex,
    x0: Mps,
    *,
    max_sweeps: int = 1,
    krylov_dimension: int = 8,
    local_tolerance: float = 1.0e-1,
    symmetry_block_bond_cap: int = 32,
    use_block_env: bool = True,
) -> tuple[Mps, AMEnSolveReport]:
    """Run short fixed-cap GMRES sweeps for an outer preconditioner.

    The supplied SRC state initializes a per-QN bounded MPS manifold.
    Enrichment is disabled; exact complementary directions may fill an
    underfull charge block up to the cap, while fixed per-QN basis selection
    makes canonical splits lossless up to roundoff. Consequently every lossy
    rank selection remains in the caller's SRC construction.
    """

    return solve_shifted_amen(
        hamiltonian,
        rhs,
        shift,
        x0=x0,
        max_sweeps=max_sweeps,
        local_maxiter=krylov_dimension,
        local_tolerance=local_tolerance,
        bond_cap=symmetry_block_bond_cap,
        cutoff=1.0e-12,
        global_tolerance=0.5,
        enrichment_rank=1,
        residual_replacement_interval=max(1, krylov_dimension),
        stagnation_sweeps=1,
        use_block_env=use_block_env,
        half_sweeps_per_sweep=2,
        certify_global_residual=False,
        symmetry_block_bond_cap=symmetry_block_bond_cap,
        local_solver="gmres",
    )


__all__ = [
    "AMEnSolveReport",
    "solve_shifted_amen",
    "solve_shifted_one_site_gmres",
]
