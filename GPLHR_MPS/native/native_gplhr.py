from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import time
from typing import Sequence

import numpy as np
import scipy.linalg
import scipy.optimize

from .native_tensor import (
    LOCAL_QN,
    NativeMPO,
    NativeMPS,
    cross_gram,
    diagonal_product_apply,
    exact_linear_combination,
    exact_mpo_apply,
    gram_matrix,
    gpu_parallel_map,
    norm,
    normalize,
    orthonormalize_src,
    overlap,
    pack_state_sites,
    product_mps,
    projected_matrices,
    random_mps,
    rotate_src,
    src_apply,
    src_round,
    src_sum,
    tensor_block_imaginary_diagnostics,
)
from .small_dense import metric_whitener, ordered_harmonic_qz, ordered_schur


@dataclass(frozen=True)
class NativeGPLHRConfig:
    active_roots: int = 4
    retained_roots: int = 4
    targeted_roots: int | None = None
    target: complex = -6.0
    # Conventional sector-summed virtual-bond dimension of each MPS.
    max_bond: int = 32
    max_iterations: int = 60
    residual_tolerance: float = 2.0e-6
    guard_residual_factor: float = 1.0
    gram_cutoff: float = 1.0e-11
    src_power_steps: int = 2
    src_oversampling_factor: float = 1.5
    src_additive_oversampling: int = 10
    orthogonalization_passes: int = 2
    jacobi_quadrature_terms: int = 6
    jacobi_quadrature_scale: float | None = None
    preconditioner: str = "direct_gmres"
    gmres_krylov_dimension: int = 2
    gmres_orthogonalization_passes: int = 1
    one_site_krylov_dimension: int = 8
    one_site_sweeps: int = 1
    one_site_local_tolerance: float = 1.0e-1
    adaptive_m1: bool = False
    initial_random_bond: int = 4
    initial_filter_steps: int = 0
    initial_filter_upper_bound: float = 50.0
    seed: int = 20260813
    verbose: bool = False
    plateau_iterations: int = 6
    plateau_relative_tolerance: float = 2.0e-3
    plateau_residual_factor: float = 100.0
    preserve_real_structure: bool = True
    
    require_real_ritz: bool = False
    real_eigenvalue_tolerance: float = 1.0e-10
    reproject_search_directions: bool = True
    
    final_residual_from_compressed_actions: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.active_roots <= self.retained_roots:
            raise ValueError("active_roots must not exceed retained_roots")
        targeted = self.retained_roots if self.targeted_roots is None else self.targeted_roots
        if not self.active_roots <= targeted <= self.retained_roots:
            raise ValueError("targeted_roots must lie between active and retained roots")
        if self.max_bond < 1 or self.max_iterations < 1:
            raise ValueError("max_bond and max_iterations must be positive")
        if self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive")
        if self.guard_residual_factor <= 0.0:
            raise ValueError("guard_residual_factor must be positive")
        if self.jacobi_quadrature_terms < 1:
            raise ValueError("jacobi quadrature needs at least one term")
        if self.preconditioner not in {
            "short_gmres",
            "direct_gmres",
            "jacobi",
            "one_site_gmres",
        }:
            raise ValueError(
                "preconditioner must be 'short_gmres', 'direct_gmres', "
                "'one_site_gmres', or 'jacobi'"
            )
        if self.gmres_krylov_dimension not in (1, 2, 3):
            raise ValueError("gmres_krylov_dimension must be 1, 2, or 3")
        if self.gmres_orthogonalization_passes not in (1, 2):
            raise ValueError("GMRES orthogonalization passes must be 1 or 2")
        if self.one_site_krylov_dimension < 1 or self.one_site_sweeps < 1:
            raise ValueError("one-site GMRES dimensions must be positive")
        if self.one_site_local_tolerance <= 0.0:
            raise ValueError("one-site GMRES tolerance must be positive")
        if (
            self.jacobi_quadrature_scale is not None
            and self.jacobi_quadrature_scale <= 0.0
        ):
            raise ValueError("jacobi_quadrature_scale must be positive")
        if self.plateau_iterations < 1:
            raise ValueError("plateau_iterations must be positive")
        if self.plateau_relative_tolerance <= 0.0:
            raise ValueError("plateau_relative_tolerance must be positive")
        if self.plateau_residual_factor < 1.0:
            raise ValueError("plateau_residual_factor must be at least one")
        if self.initial_filter_steps < 0:
            raise ValueError("initial_filter_steps must be nonnegative")
        if self.initial_filter_upper_bound <= self.target.real:
            raise ValueError("initial_filter_upper_bound must exceed target")
        if self.real_eigenvalue_tolerance <= 0.0:
            raise ValueError("real_eigenvalue_tolerance must be positive")
        if self.preserve_real_structure and abs(self.target.imag) > 0.0:
            raise ValueError("real-structure mode requires a real target")
        if self.require_real_ritz and not self.preserve_real_structure:
            raise ValueError("real-Ritz mode requires preserve_real_structure")
        if self.require_real_ritz and abs(self.target.imag) > 0.0:
            raise ValueError("real-Ritz mode requires a real target")


@dataclass
class NativeGPLHRResult:
    states: list[NativeMPS]
    actions: list[NativeMPS]
    values: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    iterations: int
    history: list[dict]
    diagnostics: dict


def _block_matrices(
    states: Sequence[NativeMPS], operator: NativeMPO
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return G, K, and L by direct MPO network contractions.

    In particular, ``L`` is contracted as ``<z_i|H^dagger H|z_j>`` without
    constructing either ``H^dagger H`` or a harmonic-test MPS block.  This
    deliberately does not reuse compressed ``H|z_j>`` objects.
    """

    gram, projected, action_gram = projected_matrices(
        states, operator, include_l=True
    )
    if action_gram is None:
        raise RuntimeError("direct shifted-image Gram contraction was skipped")
    return gram, projected, action_gram


def _block_matrices_from_actions(
    states: Sequence[NativeMPS], actions: Sequence[NativeMPS]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GPLHR matrices from the persistent compressed ``H Z`` block."""

    if len(states) != len(actions):
        raise ValueError("state and action blocks have different sizes")
    gram = gram_matrix(states)
    projected = cross_gram(states, actions)
    action_gram = gram_matrix(actions)
    return gram, projected, action_gram


def _residual_gram(
    gram: np.ndarray,
    projected: np.ndarray,
    action_gram: np.ndarray,
    action_coefficients: np.ndarray,
    model_coefficients: np.ndarray,
) -> np.ndarray:
    """Gram matrix of ``H V A - V B`` from tensor-network contractions."""

    value = (
        action_coefficients.conj().T @ action_gram @ action_coefficients
        - action_coefficients.conj().T @ projected.conj().T @ model_coefficients
        - model_coefficients.conj().T @ projected @ action_coefficients
        + model_coefficients.conj().T @ gram @ model_coefficients
    )
    return 0.5 * (value + value.conj().T)


def _residual_norms(residual_gram: np.ndarray) -> np.ndarray:
    diagonal = np.real(np.diag(residual_gram))
    scale = max(float(np.linalg.norm(residual_gram)), 1.0)
    if float(np.min(diagonal)) < -1.0e-10 * scale:
        raise FloatingPointError("contracted residual Gram matrix is indefinite")
    return np.sqrt(np.maximum(diagonal, 0.0))


def _ritz_eigenpair_residuals(
    gram: np.ndarray,
    projected: np.ndarray,
    action_gram: np.ndarray,
    transform: np.ndarray,
    triangular: np.ndarray,
    active_roots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return value-paired Ritz residuals in ascending eigenvalue order.

    The GPLHR recurrence uses an ordered Schur basis, whose individual column
    residuals are not eigenvector residuals and need not share the ordering of
    the eigenvalues of the active triangular block.  For diagnostics, rotate
    that small coefficient-space block to its right Ritz eigenvectors and
    evaluate ``||H x_i - lambda_i x_i||`` from the existing tensor-network
    contraction Grams.  No Hilbert-space vector is formed here.
    """

    active_triangular = triangular[:active_roots, :active_roots]
    values, right = scipy.linalg.eig(
        active_triangular, right=True, left=False, check_finite=False
    )
    coefficients = transform[:, :active_roots] @ right

    coefficient_gram = coefficients.conj().T @ gram @ coefficients
    coefficient_norm_squared = np.real(np.diag(coefficient_gram))
    scale = max(float(np.linalg.norm(coefficient_gram)), 1.0)
    if float(np.min(coefficient_norm_squared)) <= 1.0e-14 * scale:
        raise FloatingPointError("Ritz eigenvector has negligible MPS norm")
    coefficients = coefficients / np.sqrt(coefficient_norm_squared)[None, :]

    # ``np.lexsort`` uses the last key as primary: real part first, imaginary
    # part second.  In real-Ritz production runs this is simply low to high.
    order = np.lexsort((np.imag(values), np.real(values)))
    values = values[order]
    coefficients = coefficients[:, order]
    model_coefficients = coefficients * values[None, :]
    residual_gram = _residual_gram(
        gram,
        projected,
        action_gram,
        coefficients,
        model_coefficients,
    )
    return values, _residual_norms(residual_gram)


def _schur_analysis(
    states: Sequence[NativeMPS],
    operator: NativeMPO,
    config: NativeGPLHRConfig,
    *,
    actions: Sequence[NativeMPS] | None = None,
) -> dict:
    gram, projected, action_gram = (
        _block_matrices(states, operator)
        if actions is None
        else _block_matrices_from_actions(states, actions)
    )
    whitening = metric_whitener(gram, config.gram_cutoff)
    if whitening.retained != len(states):
        raise FloatingPointError(
            f"retained MPS block has rank {whitening.retained}, expected {len(states)}"
        )
    reduced = whitening.transform.conj().T @ projected @ whitening.transform
    schur_vectors, triangular, schur_info = ordered_schur(
        reduced,
        config.target,
        config.active_roots,
        require_real=config.require_real_ritz,
    )
    transform = whitening.transform @ schur_vectors
    active_action = transform[:, : config.active_roots]
    active_model = transform @ triangular[:, : config.active_roots]
    residual_gram = _residual_gram(
        gram, projected, action_gram, active_action, active_model
    )
    residuals = _residual_norms(residual_gram)
    all_residual_gram = _residual_gram(
        gram,
        projected,
        action_gram,
        transform,
        transform @ triangular,
    )
    all_residuals = _residual_norms(all_residual_gram)
    eigenvalues, eigenvector_residuals = _ritz_eigenpair_residuals(
        gram,
        projected,
        action_gram,
        transform,
        triangular,
        config.active_roots,
    )
    remaining = np.arange(config.active_roots, len(states), dtype=int)
    if remaining.size:
        nearest_guard = int(
            remaining[
                np.argmin(
                    np.abs(np.diag(triangular)[remaining] - config.target)
                )
            ]
        )
        nearest_guard_residual = float(all_residuals[nearest_guard])
    else:
        nearest_guard = -1
        nearest_guard_residual = 0.0
    targeted_roots = (
        config.retained_roots
        if config.targeted_roots is None
        else config.targeted_roots
    )
    active_values = scipy.linalg.eigvals(
        triangular[: config.active_roots, : config.active_roots],
        check_finite=False,
    )
    active_values = active_values[
        np.argsort(np.abs(active_values - config.target))
    ]
    return {
        "gram": gram,
        "projected": projected,
        "action_gram": action_gram,
        "whitening": whitening,
        "transform": transform,
        "triangular": triangular,
        "values": active_values,
        "residual_gram": residual_gram,
        "residual_norms": residuals,
        "maximum_residual": float(np.max(residuals)),
        "eigenvalues": eigenvalues,
        "eigenvector_residual_norms": eigenvector_residuals,
        "maximum_eigenvector_residual": float(
            np.max(eigenvector_residuals)
        ),
        "eigenpair_residual_order": "ascending real then imaginary eigenvalue",
        "all_schur_residual_norms": all_residuals,
        "nearest_guard_index": nearest_guard,
        "nearest_guard_residual": nearest_guard_residual,
        "correction_column_indices": list(range(targeted_roots)),
        "gram_error": float(np.linalg.norm(gram - np.eye(len(states)))),
        "schur": schur_info,
        "matrix_source": (
            "direct MPO contractions"
            if actions is None
            else "persistent compressed H|state> block"
        ),
    }


def _qfree_srr_analysis(
    states: Sequence[NativeMPS],
    operator: NativeMPO,
    config: NativeGPLHRConfig,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    *,
    actions: Sequence[NativeMPS] | None = None,
) -> dict:
    """Analyze the standard GPLHR harmonic Q-free Schur relation.

    Standard GPLHR carries ``A V M_B - V M_A`` from the harmonic QZ/SRR
    extraction.  Replacing these factors by a fresh eigendecomposition of
    ``V^* A V`` is ordinary Rayleigh--Ritz extraction and can create spurious
    conjugate Ritz pairs for a real-spectrum nonnormal operator.
    """

    size = len(states)
    if factor_a.shape != (size, size) or factor_b.shape != (size, size):
        raise ValueError("Q-free SRR factors and retained MPS block disagree")
    gram, projected, action_gram = (
        _block_matrices(states, operator)
        if actions is None
        else _block_matrices_from_actions(states, actions)
    )
    whitening = metric_whitener(gram, config.gram_cutoff)
    if whitening.retained != size:
        raise FloatingPointError(
            f"retained MPS block has rank {whitening.retained}, expected {size}"
        )

    active = config.active_roots
    targeted = (
        config.retained_roots
        if config.targeted_roots is None
        else config.targeted_roots
    )
    action_coefficients = factor_b[:, :targeted]
    model_coefficients = factor_a[:, :targeted]
    residual_gram = _residual_gram(
        gram,
        projected,
        action_gram,
        action_coefficients[:, :active],
        model_coefficients[:, :active],
    )
    residuals = _residual_norms(residual_gram)
    all_residual_gram = _residual_gram(
        gram, projected, action_gram, factor_b, factor_a
    )
    all_residuals = _residual_norms(all_residual_gram)

    active_a = factor_a[:active, :active]
    active_b = factor_b[:active, :active]
    factor_values, factor_vectors = scipy.linalg.eig(
        active_a,
        active_b,
        right=True,
        left=False,
        check_finite=False,
    )
    
    eigen_coefficients = factor_b[:, :active] @ factor_vectors
    coefficient_gram = eigen_coefficients.conj().T @ gram @ eigen_coefficients
    coefficient_norms = np.real(np.diag(coefficient_gram))
    if float(np.min(coefficient_norms)) <= 1.0e-14 * max(
        float(np.linalg.norm(coefficient_gram)), 1.0
    ):
        raise FloatingPointError("Q-free Ritz eigenvector has negligible MPS norm")
    eigen_coefficients /= np.sqrt(coefficient_norms)[None, :]
    order = np.lexsort((np.imag(factor_values), np.real(factor_values)))
    eigenvalues = factor_values[order]
    eigen_coefficients = eigen_coefficients[:, order]
    eigen_residual_gram = _residual_gram(
        gram,
        projected,
        action_gram,
        eigen_coefficients,
        eigen_coefficients * eigenvalues[None, :],
    )
    eigenvector_residuals = _residual_norms(eigen_residual_gram)

    
    effective_schur = scipy.linalg.solve(
        factor_b.T,
        factor_a.T,
        assume_a="gen",
        check_finite=False,
    ).T
    values_near_target = scipy.linalg.eigvals(
        active_a, active_b, check_finite=False
    )
    values_near_target = values_near_target[
        np.argsort(np.abs(values_near_target - config.target))
    ]
    return {
        "gram": gram,
        "projected": projected,
        "action_gram": action_gram,
        "whitening": whitening,
        "transform": np.eye(size, dtype=effective_schur.dtype),
        "triangular": effective_schur,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "action_coefficients": action_coefficients,
        "model_coefficients": model_coefficients,
        "values": values_near_target,
        "residual_gram": residual_gram,
        "residual_norms": residuals,
        "maximum_residual": float(np.max(residuals)),
        "eigenvalues": eigenvalues,
        
        "eigenvector_coefficients": eigen_coefficients,
        "eigenvector_residual_norms": eigenvector_residuals,
        "maximum_eigenvector_residual": float(np.max(eigenvector_residuals)),
        "eigenpair_residual_order": "ascending real then imaginary eigenvalue",
        "all_schur_residual_norms": all_residuals,
        "nearest_guard_index": -1,
        "nearest_guard_residual": 0.0,
        "correction_column_indices": list(range(targeted)),
        "gram_error": float(np.linalg.norm(gram - np.eye(size))),
        "schur": {
            "dimension": size,
            "active_dimension": active,
            "schur_form": "harmonic_qfree_srr",
            "selection_mode": "standard_gplhr_harmonic_qz_factors",
            "real_1x1_ritz_constraint": False,
            "selected_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in values_near_target
            ],
        },
        "matrix_source": (
            "direct MPO contractions"
            if actions is None
            else "persistent compressed H|state> block"
        ),
    }


def _maximum_overlap_eigenpair_order(
    previous_coefficients: np.ndarray,
    cross_basis_gram: np.ndarray,
    current_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-to-one current-root order maximizing physical overlap.

    For old and new physical Ritz vectors, their overlap matrix is the old
    coefficients adjoint times the cross-basis Gram times the new
    coefficients. Absolute overlaps remove the arbitrary complex phase of
    every Ritz vector. The Hungarian assignment prevents two old roots from
    being matched to the same new vector.
    """

    previous_coefficients = np.asarray(previous_coefficients)
    cross_basis_gram = np.asarray(cross_basis_gram)
    current_coefficients = np.asarray(current_coefficients)
    if previous_coefficients.ndim != 2 or current_coefficients.ndim != 2:
        raise ValueError("Ritz coefficient arrays must be matrices")
    if cross_basis_gram.shape != (
        previous_coefficients.shape[0],
        current_coefficients.shape[0],
    ):
        raise ValueError("cross-basis Gram matrix has incompatible dimensions")
    if previous_coefficients.shape[1] != current_coefficients.shape[1]:
        raise ValueError("consecutive Ritz blocks have different root counts")

    absolute_overlaps = np.abs(
        previous_coefficients.conj().T
        @ cross_basis_gram
        @ current_coefficients
    )
    old_indices, new_indices = scipy.optimize.linear_sum_assignment(
        -absolute_overlaps
    )
    order = np.empty(previous_coefficients.shape[1], dtype=int)
    order[old_indices] = new_indices
    matched = absolute_overlaps[np.arange(order.size), order]
    return order, matched


def _align_eigenpair_diagnostics(
    previous_states: Sequence[NativeMPS],
    previous_analysis: dict,
    current_states: Sequence[NativeMPS],
    current_analysis: dict,
) -> dict:
    """Relabel current eigenpair diagnostics by consecutive MPS overlaps."""

    previous_coefficients = previous_analysis["eigenvector_coefficients"]
    current_coefficients = current_analysis["eigenvector_coefficients"]
    cross_basis_gram = cross_gram(previous_states, current_states)
    order, matched = _maximum_overlap_eigenpair_order(
        previous_coefficients,
        cross_basis_gram,
        current_coefficients,
    )
    current_analysis["eigenvalues"] = current_analysis["eigenvalues"][order]
    current_analysis["eigenvector_coefficients"] = current_coefficients[:, order]
    current_analysis["eigenvector_residual_norms"] = current_analysis[
        "eigenvector_residual_norms"
    ][order]
    current_analysis["eigenpair_residual_order"] = (
        "persistent root identity by maximum absolute consecutive physical "
        "MPS overlap (Hungarian one-to-one assignment)"
    )
    current_analysis["eigenpair_tracking"] = {
        "method": "maximum_absolute_consecutive_physical_mps_overlap",
        "assignment": "Hungarian one-to-one",
        "raw_current_indices_by_persistent_root": [
            int(index) for index in order
        ],
        "matched_absolute_overlaps": [float(value) for value in matched],
        "minimum_matched_absolute_overlap": float(np.min(matched)),
    }
    return current_analysis


def _materialize_residuals(
    states: Sequence[NativeMPS],
    actions: Sequence[NativeMPS],
    operator: NativeMPO,
    analysis: dict,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[list[NativeMPS], list[dict]]:
    correction_columns = analysis["correction_column_indices"]
    action_coefficients = analysis.get("action_coefficients")
    model_coefficients = analysis.get("model_coefficients")
    if action_coefficients is None or model_coefficients is None:
        action_coefficients = analysis["transform"][:, correction_columns]
        model_coefficients = (
            analysis["transform"]
            @ analysis["triangular"][:, correction_columns]
        )
    def materialize(root):
        started = time.perf_counter()
        exact_action = exact_linear_combination(
            actions, action_coefficients[:, root]
        )
        model_state = exact_linear_combination(
            states, model_coefficients[:, root]
        )
        exact_residual = exact_linear_combination(
            [exact_action, model_state], [1.0, -1.0]
        )
        del exact_action, model_state
        state, report = src_round(
            exact_residual,
            config.max_bond,
            seed=config.seed + 104729 * call_index + 1009 * root,
            power_steps=config.src_power_steps,
            consume_input=True,
        )
        report["operation"] = "symbolic_schur_residual_mpo_mps"
        residual_norm = norm(state)
        retained_overlaps = cross_gram(states, [state])[:, 0]
        report["post_src_retained_overlap_ratio"] = float(
            np.linalg.norm(retained_overlaps)
            / max(residual_norm, 1.0e-300)
        )
        report["residual_output_seconds"] = time.perf_counter() - started
        state.validate(config.max_bond)
        return state, report

    materialized = gpu_parallel_map(materialize, range(len(correction_columns)))
    residuals = [item[0] for item in materialized]
    reports = [item[1] for item in materialized]
    return residuals, reports


def _materialize_arnoldi_residuals(
    corrections: Sequence[NativeMPS],
    operator: NativeMPO,
    factor_a_active: np.ndarray,
    factor_b_active: np.ndarray,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[list[NativeMPS], list[dict]]:
    """Materialize ``H W M_B - W M_A`` with one final SRC per column."""

    expected = (len(corrections), len(corrections))
    if factor_a_active.shape != expected or factor_b_active.shape != expected:
        raise ValueError("active Q-free Schur factors and W have incompatible sizes")
    output = []
    reports = []
    for column, correction in enumerate(corrections):
        started = time.perf_counter()
        action_input = exact_linear_combination(
            corrections, factor_b_active[:, column]
        )
        exact_action = exact_mpo_apply(operator, action_input)
        del action_input
        model = exact_linear_combination(
            corrections, factor_a_active[:, column]
        )
        exact_residual = exact_linear_combination(
            [exact_action, model], [1.0, -1.0]
        )
        del exact_action, model
        residual, report = src_round(
            exact_residual,
            config.max_bond,
            seed=config.seed + 104729 * call_index + 1009 * column,
            power_steps=config.src_power_steps,
            consume_input=True,
        )
        report["operation"] = "adaptive_arnoldi_residual_mpo_mps"
        report["residual_output_seconds"] = time.perf_counter() - started
        output.append(residual)
        reports.append(report)
    return output, reports


def _project_src(
    state: NativeMPS,
    basis: Sequence[NativeMPS],
    config: NativeGPLHRConfig,
    *,
    call_index: int,
    basis_gram: np.ndarray | None = None,
    basis_packed=None,
    passes: int | None = None,
) -> tuple[NativeMPS | None, dict]:
    candidate = state.copy()
    initial_norm = norm(candidate)
    reports = []
    if basis_packed is None and basis:
        basis_packed = pack_state_sites(basis)
    if basis_gram is None and basis:
        basis_gram = (
            cross_gram(
                basis,
                basis,
                left_packed=basis_packed,
                right_packed=basis_packed,
            )
            if basis_packed is not None
            else gram_matrix(basis)
        )
        basis_gram = 0.5 * (basis_gram + basis_gram.conj().T)
    if basis_gram is not None and basis_gram.shape != (len(basis), len(basis)):
        raise ValueError("cached projection Gram matrix has the wrong shape")
    projection_passes = config.orthogonalization_passes if passes is None else passes
    if projection_passes < 1:
        raise ValueError("projection passes must be positive")
    for orthogonalization_pass in range(projection_passes):
        if not basis:
            break
        overlaps = cross_gram(
            basis, [candidate], left_packed=basis_packed
        )[:, 0]
        coefficients = (
            -np.linalg.pinv(basis_gram, rcond=config.gram_cutoff) @ overlaps
        )
        candidate, report = src_sum(
            [candidate, *basis],
            [1.0, *coefficients],
            config.max_bond,
            seed=(
                config.seed
                + 104729 * call_index
                + 1009 * orthogonalization_pass
            ),
            power_steps=config.src_power_steps,
        )
        reports.append({"pass": orthogonalization_pass, **report})
    final_norm = norm(candidate)
    if final_norm <= config.gram_cutoff * max(initial_norm, 1.0e-300):
        return None, {
            "initial_norm": initial_norm,
            "final_norm": final_norm,
            "dropped": True,
            "src_projections": reports,
        }
    candidate = candidate.scale(1.0 / final_norm)
    maximum_overlap = (
        float(
            np.max(
                np.abs(
                    cross_gram(
                        basis,
                        [candidate],
                        left_packed=basis_packed,
                    )[:, 0]
                )
            )
        )
        if basis
        else 0.0
    )
    return candidate, {
        "initial_norm": initial_norm,
        "final_norm": final_norm,
        "dropped": False,
        "maximum_post_projection_overlap": float(maximum_overlap),
        "src_projections": reports,
    }


def _local_diagonal_energies(parameters) -> list[np.ndarray]:
    output = []
    for potential in parameters.onsite_potential:
        row = []
        for n_up, n_down in LOCAL_QN:
            row.append(
                potential * (n_up + n_down)
                + parameters.interaction * n_up * n_down
            )
        output.append(np.asarray(row, dtype=np.float64))
    return output


def _jacobi_precondition_src(
    state: NativeMPS,
    parameters,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[NativeMPS, dict]:
    """Apply an exponential-sum approximation to ``(D-target I)^-1``.

    ``D`` is the diagonal Hubbard MPO.  Each exponential is an exact bond-one
    diagonal product operator.  Their sum is rounded with symmetry-resolved
    SRC, like every other lossy linear combination in this implementation.
    """

    site_energies = _local_diagonal_energies(parameters)
    diagonal_min = sum(float(np.min(row)) for row in site_energies)
    diagonal_max = sum(float(np.max(row)) for row in site_energies)
    shifted_min = diagonal_min - config.target.real
    shifted_max = diagonal_max - config.target.real
    if shifted_min <= 0.0:
        raise ValueError(
            "exponential Jacobi inverse requires target below the diagonal range"
        )
    scale = config.jacobi_quadrature_scale or math.sqrt(
        shifted_min * shifted_max
    )
    nodes, weights = np.polynomial.laguerre.laggauss(
        config.jacobi_quadrature_terms
    )
    times = nodes / scale
    coefficients = (
        weights / scale * np.exp(nodes + config.target * times)
    )
    started = time.perf_counter()
    terms_started = time.perf_counter()
    terms = []
    for quadrature_time in times:
        local_diagonals = [
            np.exp(-quadrature_time * energies) for energies in site_energies
        ]
        terms.append(diagonal_product_apply(state, local_diagonals))
    term_construction_seconds = time.perf_counter() - terms_started
    addition_started = time.perf_counter()
    output, compression = src_sum(
        terms,
        coefficients,
        config.max_bond,
        seed=config.seed + 104729 * call_index,
        power_steps=config.src_power_steps,
    )
    addition_seconds = time.perf_counter() - addition_started
    samples = np.linspace(diagonal_min, diagonal_max, 257) - config.target.real
    samples = samples[samples > 0.0]
    approximation = np.sum(
        (weights * np.exp(nodes) / scale)[:, None]
        * np.exp(-times[:, None] * samples[None, :]),
        axis=0,
    )
    exact = 1.0 / samples
    return output, {
        "method": (
            "scaled Gauss-Laguerre bond-one exponential-sum diagonal "
            "Jacobi + SRC addition"
        ),
        "quadrature_terms": config.jacobi_quadrature_terms,
        "quadrature_scale": scale,
        "sampled_maximum_relative_inverse_error": float(
            np.max(np.abs(approximation - exact) / exact)
        ),
        "term_construction_seconds": term_construction_seconds,
        "src_addition_seconds": addition_seconds,
        "total_seconds": time.perf_counter() - started,
        "compression": compression,
        "mpo_actions": 0,
        "src_linear_combinations": 1,
    }


def _shifted_apply_src(
    operator: NativeMPO,
    state: NativeMPS,
    config: NativeGPLHRConfig,
    *,
    seed: int,
) -> tuple[NativeMPS, dict]:
    """Apply ``H-target*I`` as one symbolic expression and one SRC."""

    exact_action = exact_mpo_apply(operator, state)
    exact_shifted = exact_linear_combination(
        [exact_action, state], [1.0, -config.target]
    )
    del exact_action
    output, report = src_round(
        exact_shifted,
        config.max_bond,
        seed=seed,
        power_steps=config.src_power_steps,
        consume_input=True,
    )
    report["operation"] = "shifted_mpo_mps_action"
    return output, report


def _short_gmres_precondition_src(
    source: NativeMPS,
    operator: NativeMPO,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[NativeMPS, dict]:
    """Cheap global GMRES polynomial using only MPO--MPS/SRC operations.

    The temporary Krylov vectors are discarded immediately after the inexact
    solve, so no recurrence crosses an outer GPLHR restart.
    """

    started = time.perf_counter()
    beta = norm(source)
    if beta == 0.0:
        raise FloatingPointError("cannot precondition a zero residual")
    dimension = config.gmres_krylov_dimension
    basis = [source.scale(1.0 / beta)]
    hessenberg = np.zeros((dimension + 1, dimension), dtype=np.complex128)
    action_reports = []
    orthogonalization_reports = []
    for column in range(dimension):
        candidate, action_report = _shifted_apply_src(
            operator,
            basis[column],
            config,
            seed=(
                config.seed
                + 104729 * call_index
                + 1009 * column
            ),
        )
        action_reports.append(action_report)
        pass_reports = []
        for orthogonalization_pass in range(
            config.gmres_orthogonalization_passes
        ):
            coefficients = cross_gram(basis, [candidate])[:, 0]
            hessenberg[: len(basis), column] += coefficients
            candidate, report = src_sum(
                [candidate, *basis],
                [1.0, *(-coefficients)],
                config.max_bond,
                seed=(
                    config.seed
                    + 104729 * call_index
                    + 1009 * column
                    + 37 * (orthogonalization_pass + 1)
                ),
                power_steps=config.src_power_steps,
            )
            pass_reports.append(report)
        next_norm = norm(candidate)
        hessenberg[column + 1, column] = next_norm
        orthogonalization_reports.append(pass_reports)
        if column + 1 < dimension:
            if next_norm <= config.gram_cutoff * beta:
                dimension = column + 1
                hessenberg = hessenberg[: dimension + 1, :dimension]
                break
            basis.append(candidate.scale(1.0 / next_norm))

    right_hand_side = np.zeros(dimension + 1, dtype=np.complex128)
    right_hand_side[0] = beta
    coefficients, _, _, _ = scipy.linalg.lstsq(
        hessenberg[: dimension + 1, :dimension],
        right_hand_side,
        check_finite=False,
        lapack_driver="gelsy",
    )
    output, output_report = src_sum(
        basis[:dimension],
        coefficients,
        config.max_bond,
        seed=config.seed + 104729 * call_index + 900001,
        power_steps=config.src_power_steps,
    )
    small_residual = np.linalg.norm(
        right_hand_side
        - hessenberg[: dimension + 1, :dimension] @ coefficients
    ) / beta
    return output, {
        "method": "temporary global MPO--MPS GMRES with SRC",
        "requested_krylov_dimension": config.gmres_krylov_dimension,
        "orthogonalization_passes": config.gmres_orthogonalization_passes,
        "achieved_krylov_dimension": dimension,
        "small_relative_residual": float(small_residual),
        "hessenberg": hessenberg.tolist(),
        "shifted_actions": action_reports,
        "orthogonalizations": orthogonalization_reports,
        "output_compression": output_report,
        "mpo_actions": dimension,
        "src_linear_combinations": (
            dimension * config.gmres_orthogonalization_passes + 1
        ),
        "total_seconds": time.perf_counter() - started,
    }


def _direct_gmres_precondition_src(
    source: NativeMPS,
    operator: NativeMPO,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[NativeMPS, dict]:
    """Minimum residual in the GMRES power-Krylov space without Arnoldi.

    For dimension two this spans exactly ``{r, (H-target I)r}``, the same
    correction space as short-GMRES(2).  The minimum-residual coefficients are
    recovered from native MPO/MPS contractions.  Avoiding lossy Arnoldi
    orthogonalizations removes two SRC additions while preserving the GMRES
    subspace and its residual minimization problem.
    """

    started = time.perf_counter()
    beta = norm(source)
    if beta == 0.0:
        raise FloatingPointError("cannot precondition a zero residual")
    requested_dimension = config.gmres_krylov_dimension
    basis = [source.scale(1.0 / beta)]
    actions = []
    action_reports = []
    power_normalizations = []
    for column in range(requested_dimension):
        action, action_report = _shifted_apply_src(
            operator,
            basis[column],
            config,
            seed=config.seed + 104729 * call_index + 1009 * column,
        )
        actions.append(action)
        action_reports.append(action_report)
        if column + 1 < requested_dimension:
            next_norm = norm(action)
            power_normalizations.append(float(next_norm))
            if next_norm <= config.gram_cutoff:
                break
            basis.append(action.scale(1.0 / next_norm))

    dimension = len(actions)
    action_gram = cross_gram(actions, actions)
    action_gram = 0.5 * (action_gram + action_gram.conj().T)
    right_hand_side = cross_gram(actions, [source])[:, 0]
    whitening = metric_whitener(action_gram, config.gram_cutoff)
    coefficients = whitening.transform @ (
        whitening.transform.conj().T @ right_hand_side
    )
    output, output_report = src_sum(
        basis[:dimension],
        coefficients,
        config.max_bond,
        seed=config.seed + 104729 * call_index + 900001,
        power_steps=config.src_power_steps,
    )
    residual_squared = (
        beta * beta
        - 2.0 * np.real(np.vdot(coefficients, right_hand_side))
        + np.real(np.vdot(coefficients, action_gram @ coefficients))
    )
    residual_squared = max(float(residual_squared), 0.0)
    return output, {
        "method": (
            "direct minimum-residual power-Krylov GMRES with native "
            "MPO/MPS contractions and SRC"
        ),
        "requested_krylov_dimension": requested_dimension,
        "achieved_krylov_dimension": dimension,
        "retained_normal_equation_rank": whitening.retained,
        "normal_equation_condition_number": whitening.condition_number,
        "small_relative_residual": math.sqrt(residual_squared) / beta,
        "power_normalizations": power_normalizations,
        "shifted_actions": action_reports,
        "output_compression": output_report,
        "mpo_actions": dimension,
        "src_linear_combinations": 1,
        "total_seconds": time.perf_counter() - started,
    }


def _apply_preconditioner_src(
    state: NativeMPS,
    operator: NativeMPO,
    parameters,
    config: NativeGPLHRConfig,
    *,
    call_index: int,
    one_site_preconditioner=None,
) -> tuple[NativeMPS, dict]:
    if config.preconditioner == "one_site_gmres":
        if one_site_preconditioner is None:
            raise RuntimeError("one-site GMRES preconditioner was not initialized")
        return one_site_preconditioner.apply(state, call_index=call_index)
    if config.preconditioner == "short_gmres":
        return _short_gmres_precondition_src(
            state, operator, config, call_index=call_index
        )
    if config.preconditioner == "direct_gmres":
        return _direct_gmres_precondition_src(
            state, operator, config, call_index=call_index
        )
    return _jacobi_precondition_src(
        state, parameters, config, call_index=call_index
    )


class _NativeOneSiteGMRESPreconditioner:
    """One-site MPS sweep approximation to ``(H-target I)^-1``.

    Rank enrichment and deterministic truncation are disabled.  Exact local
    gauge moves can expose additional linearly independent charge-sector
    rows, so the converted result is always returned through one explicit
    native SRC round to enforce the conventional sector-summed cap.
    """

    def __init__(self, parameters, config: NativeGPLHRConfig):
        from .amen_preconditioner import build_hubbard_mpo

        self.parameters = parameters
        self.config = config
        self.model, self.hamiltonian = build_hubbard_mpo(
            parameters, similarity_balanced=True
        )

    def apply(self, state: NativeMPS, *, call_index: int):
        from projects.feast_mps_hubbard_nonhermitian.amen_solver import (
            solve_shifted_amen,
        )
        from .amen_preconditioner import (
            apply_hubbard_similarity,
            native_src_to_renormalizer,
            renormalizer_to_native,
        )

        started = time.perf_counter()
        physical_source = native_src_to_renormalizer(state, self.model)
        balanced_source = apply_hubbard_similarity(
            physical_source, self.parameters, inverse=True
        )
        rhs = balanced_source.copy()
        rhs.scale(-1.0, inplace=True)
        initial = balanced_source.copy()
        initial.scale(
            1.0 / max(abs(complex(self.config.target)), 1.0), inplace=True
        )
        solution, solve_report = solve_shifted_amen(
            self.hamiltonian,
            rhs,
            complex(self.config.target),
            x0=initial,
            max_sweeps=self.config.one_site_sweeps,
            local_maxiter=self.config.one_site_krylov_dimension,
            local_tolerance=self.config.one_site_local_tolerance,
            bond_cap=self.config.max_bond,
            cutoff=1.0e-12,
            global_tolerance=0.5,
            enrichment_rank=0,
            residual_replacement_interval=max(
                1, self.config.one_site_krylov_dimension
            ),
            stagnation_sweeps=1,
            use_block_env=True,
            half_sweeps_per_sweep=2,
            certify_global_residual=False,
            symmetry_block_bond_cap=state.max_charge_block_bond,
            local_solver="gmres",
        )
        physical_solution = apply_hubbard_similarity(
            solution, self.parameters, inverse=False
        )
        unrounded = renormalizer_to_native(physical_solution)
        transient_maximum_bond = unrounded.max_bond
        output, compression = src_round(
            unrounded,
            self.config.max_bond,
            seed=self.config.seed + 104729 * call_index + 1700003,
            power_steps=self.config.src_power_steps,
            oversampling_factor=self.config.src_oversampling_factor,
            additive_oversampling=self.config.src_additive_oversampling,
            consume_input=True,
        )
        output.validate(self.config.max_bond)
        return output, {
            "method": (
                "one-site MPS GMRES sweep in the exactly similarity-balanced "
                "Hubbard frame followed by native sector-summed SRC"
            ),
            "call_index": call_index,
            "sweeps": solve_report.sweeps,
            "half_sweeps": solve_report.half_sweeps,
            "local_krylov_dimension": self.config.one_site_krylov_dimension,
            "local_iterations": solve_report.local_iterations,
            "local_matvecs": solve_report.local_matvecs,
            "local_failures": solve_report.local_failures,
            "maximum_local_relative_residual": (
                solve_report.max_local_relative_residual
            ),
            "input_maximum_bond": state.max_bond,
            "exact_sweep_transient_maximum_bond": transient_maximum_bond,
            "output_maximum_bond": output.max_bond,
            "rank_enrichment": 0,
            "deterministic_lossy_truncations": 0,
            "output_src": compression,
            "mpo_actions": 0,
            "src_linear_combinations": 1,
            "total_seconds": time.perf_counter() - started,
        }


def _apply_block_src(
    operator: NativeMPO,
    states: Sequence[NativeMPS],
    config: NativeGPLHRConfig,
    *,
    call_index: int,
) -> tuple[list[NativeMPS], list[dict], float]:
    started = time.perf_counter()
    def apply_column(item):
        column, state = item
        column_started = time.perf_counter()
        action, report = src_apply(
            operator,
            state,
            config.max_bond,
            seed=config.seed + 104729 * call_index + 1009 * column,
            power_steps=config.src_power_steps,
            oversampling_factor=config.src_oversampling_factor,
            additive_oversampling=config.src_additive_oversampling,
        )
        if report.get("operation") != "mpo_mps_action":
            raise RuntimeError("H|psi> did not use the SRC action path")
        report["action_output_seconds"] = time.perf_counter() - column_started
        action.validate(config.max_bond)
        return action, report

    applied = gpu_parallel_map(apply_column, list(enumerate(states)))
    output = [item[0] for item in applied]
    reports = [item[1] for item in applied]
    return output, reports, time.perf_counter() - started


def _harmonic_restart(
    states: Sequence[NativeMPS],
    operator: NativeMPO,
    config: NativeGPLHRConfig,
    *,
    actions: Sequence[NativeMPS] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    gram, projected, action_gram = (
        _block_matrices(states, operator)
        if actions is None
        else _block_matrices_from_actions(states, actions)
    )
    whitening = metric_whitener(gram, config.gram_cutoff)
    if whitening.retained < config.retained_roots:
        raise FloatingPointError(
            "trial Gram rank is smaller than the requested restart block"
        )
    cg = whitening.transform
    shifted = (
        action_gram
        - config.target * projected.conj().T
        - complex(config.target).conjugate() * projected
        + abs(config.target) ** 2 * gram
    )
    shifted = 0.5 * (shifted + shifted.conj().T)
    cross = projected.conj().T - complex(config.target).conjugate() * gram
    m_hat = cg.conj().T @ shifted @ cg
    m_hat = 0.5 * (m_hat + m_hat.conj().T)
    n_hat = cg.conj().T @ cross @ cg
    shifted_values, shifted_vectors = scipy.linalg.eigh(
        m_hat, check_finite=False
    )
    largest = max(float(shifted_values[-1]), 0.0)
    keep = shifted_values > config.gram_cutoff * largest
    shifted_rank = int(np.count_nonzero(keep))
    
    selected_roots = config.retained_roots
    if shifted_rank < selected_roots:
        raise FloatingPointError(
            "shifted-image rank is smaller than the GPLHR restart block"
        )
    targeted_roots = (
        config.retained_roots
        if config.targeted_roots is None
        else config.targeted_roots
    )
    transform = shifted_vectors[:, keep]
    qz_vectors, theta, factor_a, factor_b, qz_info = ordered_harmonic_qz(
        transform.conj().T @ n_hat @ transform,
        transform.conj().T @ m_hat @ transform,
        selected_roots,
        config.target,
        required_real_roots=(
            targeted_roots if config.require_real_ritz else 0
        ),
        return_srr_factors=True,
    )
    coefficients = cg @ transform @ qz_vectors
    coefficient_metric = coefficients.conj().T @ gram @ coefficients
    coefficient_metric = 0.5 * (
        coefficient_metric + coefficient_metric.conj().T
    )
    cleanup = metric_whitener(coefficient_metric, config.gram_cutoff)
    if cleanup.retained != selected_roots:
        raise FloatingPointError("harmonic restart coefficients lost rank")
    
    cleanup_cholesky = scipy.linalg.cholesky(
        coefficient_metric, lower=False, check_finite=False
    )
    cleanup_transform = scipy.linalg.solve_triangular(
        cleanup_cholesky,
        np.eye(selected_roots, dtype=coefficient_metric.dtype),
        lower=False,
        check_finite=False,
    )
    coefficients = coefficients @ cleanup_transform
    cleanup_inverse = cleanup_cholesky
    factor_a = cleanup_inverse @ factor_a @ cleanup_transform
    factor_b = cleanup_inverse @ factor_b @ cleanup_transform

   
    search_coefficients = coefficients[:, :targeted_roots].copy()
    search_coefficients[: config.retained_roots, :] = 0.0
    search_coefficients -= coefficients @ (
        coefficients.conj().T @ gram @ search_coefficients
    )
    search_metric = search_coefficients.conj().T @ gram @ search_coefficients
    search_metric = 0.5 * (search_metric + search_metric.conj().T)
    search_eigenvalues = scipy.linalg.eigvalsh(
        search_metric, check_finite=False
    )
    search_scale = max(float(search_eigenvalues[-1]), 0.0)
    search_rank = int(
        np.count_nonzero(
            search_eigenvalues > config.gram_cutoff * search_scale
        )
    ) if search_scale > 0.0 else 0
    
    if search_rank:
        search_cleanup = metric_whitener(
            search_metric, config.gram_cutoff
        )
        search_coefficients = search_coefficients @ search_cleanup.transform
    else:
        search_coefficients = np.empty(
            (coefficients.shape[0], 0), dtype=coefficients.dtype
        )

    active_action = (
        coefficients @ factor_b[:, : config.active_roots]
    )
    active_model = (
        coefficients @ factor_a[:, : config.active_roots]
    )
    predicted_gram = _residual_gram(
        gram, projected, action_gram, active_action, active_model
    )
    predicted = _residual_norms(predicted_gram)
    return coefficients, search_coefficients, factor_a, factor_b, {
        "trial_dimension": len(states),
        "trial_rank": whitening.retained,
        "shifted_image_rank": shifted_rank,
        "selected_harmonic_dimension": selected_roots,
        "requested_search_direction_columns": targeted_roots,
        "search_direction_columns": search_rank,
        "deflated_search_direction_columns": targeted_roots - search_rank,
        "search_direction_construction": (
            "standard GPLHR(0) non-V rows of selected harmonic Schur vectors"
        ),
        "selected_harmonic_values": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in theta
        ],
        "predicted_residual_norms": [float(value) for value in predicted],
        "predicted_maximum_residual": float(np.max(predicted)),
        "qz": qz_info,
        "srr_extraction": (
            "standard harmonic Q-free factors; no post-QZ ordinary "
            "Rayleigh-Ritz eigendecomposition"
        ),
        "matrix_source": (
            "direct MPO contractions"
            if actions is None
            else "persistent compressed H|trial> block"
        ),
    }


def initial_hubbard_block(parameters, config: NativeGPLHRConfig) -> list[NativeMPS]:
    """Create a reference-free, symmetry-native initial block."""

    if parameters.n_up + parameters.n_down != parameters.sites:
        raise ValueError("the default initializer expects half filling")
    local_patterns: list[tuple[int, ...]] = []
    for phase in (0, 1):
        pattern = tuple(
            2 if (site + phase) % 2 == 0 else 1
            for site in range(parameters.sites)
        )
        if sum(LOCAL_QN[value][0] for value in pattern) == parameters.n_up:
            local_patterns.append(pattern)

    
    candidates = set(local_patterns)
    for base in tuple(local_patterns):
        up_sites = [site for site, physical in enumerate(base) if physical == 2]
        down_sites = [site for site, physical in enumerate(base) if physical == 1]
        for up_site in up_sites:
            for down_site in down_sites:
                exchanged = list(base)
                exchanged[up_site], exchanged[down_site] = (
                    exchanged[down_site], exchanged[up_site]
                )
                candidates.add(tuple(exchanged))

    def hopping_connectivity(pattern: tuple[int, ...]) -> int:
        return sum(left != right for left, right in zip(pattern, pattern[1:]))

    local_patterns = sorted(
        candidates,
        key=lambda pattern: (-hopping_connectivity(pattern), pattern),
    )

    
    domain_pattern = tuple(
        [2] * parameters.n_up + [1] * parameters.n_down
    )
    
    product_columns = min(
        config.active_roots, max(1, config.retained_roots // 2)
    )
    for shift in range(parameters.sites):
        shifted = (
            domain_pattern[shift:] + domain_pattern[:shift]
        )
        if shifted not in local_patterns:
            local_patterns.append(shifted)
        if len(local_patterns) >= product_columns:
            break
    patterns = [
        product_mps(pattern) for pattern in local_patterns[:product_columns]
    ]
    random_bond = min(config.max_bond, max(config.initial_random_bond, 1))
    while len(patterns) < config.retained_roots:
        patterns.append(
            random_mps(
                parameters.sites,
                (parameters.n_up, parameters.n_down),
                random_bond,
                seed=config.seed + 7919 * len(patterns),
            )
        )
    block, _ = orthonormalize_src(
        patterns,
        config.max_bond,
        seed=config.seed + 700001,
        power_steps=config.src_power_steps,
        rank_cutoff=config.gram_cutoff,
        passes=config.orthogonalization_passes,
    )
    if len(block) != config.retained_roots:
        raise FloatingPointError("initial MPS block lost rank")
    return block


def solve_native_gplhr(
    operator: NativeMPO,
    parameters,
    *,
    initial_states: Sequence[NativeMPS] | None = None,
    config: NativeGPLHRConfig | None = None,
) -> NativeGPLHRResult:
    """Run guarded block GPLHR(0) entirely in native MPS/MPO arithmetic."""

    config = config or NativeGPLHRConfig()
    started = time.perf_counter()
    one_site_preconditioner = (
        _NativeOneSiteGMRESPreconditioner(parameters, config)
        if config.preconditioner == "one_site_gmres"
        else None
    )
    requested_preconditioner_gpus = int(
        os.environ.get("GPLHR_PRECONDITIONER_GPUS", "1")
    )
    if requested_preconditioner_gpus < 1:
        raise ValueError("GPLHR_PRECONDITIONER_GPUS must be positive")
    multi_gpu_preconditioner = None
    if requested_preconditioner_gpus > 1:
        if one_site_preconditioner is None:
            raise ValueError(
                "multi-GPU preconditioning currently requires one_site_gmres"
            )
        from .multigpu_preconditioner import MultiGPUOneSitePreconditioner

        multi_gpu_preconditioner = MultiGPUOneSitePreconditioner(
            parameters,
            config,
            one_site_preconditioner,
            requested_preconditioner_gpus,
        )
    maximum_operator_imaginary_entry = max(
        (
            abs(entry.value.imag)
            for site in operator.entries
            for entry in site
        ),
        default=0.0,
    )
    if (
        config.preserve_real_structure
        and maximum_operator_imaginary_entry
        > config.real_eigenvalue_tolerance
    ):
        raise ValueError("real-structure mode requires a real MPO")
    states = list(initial_states or initial_hubbard_block(parameters, config))
    if len(states) != config.retained_roots:
        raise ValueError("initial block size does not match retained_roots")
    for state in states:
        state.validate(config.max_bond)
        if state.total_qn != (parameters.n_up, parameters.n_down):
            raise ValueError("initial state has the wrong total charge")
    states, initial_orthogonalization = orthonormalize_src(
        states,
        config.max_bond,
        seed=config.seed + 800001,
        power_steps=config.src_power_steps,
        rank_cutoff=config.gram_cutoff,
        passes=config.orthogonalization_passes,
    )
    if len(states) != config.retained_roots:
        raise FloatingPointError("initial orthogonalization lost block rank")

    action_call = 0
    linear_call = 0
    initial_filter_reports = []
    initial_filter_action_seconds = 0.0
    initial_filter_src_actions = 0
    initial_filter_src_additions = 0
    for filter_step in range(config.initial_filter_steps):
        filter_actions, action_reports, action_seconds = _apply_block_src(
            operator, states, config, call_index=action_call
        )
        action_call += 1
        initial_filter_action_seconds += action_seconds
        initial_filter_src_actions += len(filter_actions)
        filtered_states = []
        filter_src_reports = []
        for state, action in zip(states, filter_actions):
            filtered, report = src_sum(
                [state, action],
                [config.initial_filter_upper_bound, -1.0],
                config.max_bond,
                seed=config.seed + 104729 * linear_call,
                power_steps=config.src_power_steps,
            )
            linear_call += 1
            filtered_states.append(filtered)
            filter_src_reports.append(report)
        states, orthogonalization = orthonormalize_src(
            filtered_states,
            config.max_bond,
            seed=config.seed + 104729 * linear_call,
            power_steps=config.src_power_steps,
            rank_cutoff=config.gram_cutoff,
            passes=config.orthogonalization_passes,
        )
        linear_call += 1
        if len(states) != config.retained_roots:
            raise FloatingPointError("initial filtered MPS block lost rank")
        initial_filter_src_additions += len(filter_src_reports) + len(
            orthogonalization["src_additions"]
        )
        initial_filter_reports.append(
            {
                "step": filter_step,
                "h_actions": action_reports,
                "filter_src": filter_src_reports,
                "orthogonalization": orthogonalization,
            }
        )

    actions, initial_action_reports, initial_action_seconds = _apply_block_src(
        operator, states, config, call_index=action_call
    )
    action_call += 1
    history = []
    total_src_actions = initial_filter_src_actions + len(actions)
    total_src_additions = (
        len(initial_orthogonalization["src_additions"])
        + initial_filter_src_additions
    )
    converged = False
    initial_analysis_started = time.perf_counter()
    (
        initial_coefficients,
        _,
        factor_a,
        factor_b,
        initial_harmonic_report,
    ) = _harmonic_restart(states, operator, config, actions=actions)
    states, initial_harmonic_rotation_reports = rotate_src(
        states,
        initial_coefficients,
        config.max_bond,
        seed=config.seed + 900001,
        power_steps=config.src_power_steps,
    )
    total_src_additions += len(initial_harmonic_rotation_reports)
    actions, initial_srr_action_reports, initial_srr_action_seconds = (
        _apply_block_src(operator, states, config, call_index=action_call)
    )
    action_call += 1
    total_src_actions += len(actions)
    analysis = _qfree_srr_analysis(
        states,
        operator,
        config,
        factor_a,
        factor_b,
        actions=actions,
    )
    analysis["eigenpair_tracking"] = {
        "method": "initial_ascending_real_then_imaginary_eigenvalue",
        "assignment": "identity",
        "raw_current_indices_by_persistent_root": list(
            range(config.active_roots)
        ),
        "matched_absolute_overlaps": None,
        "minimum_matched_absolute_overlap": None,
    }
    initial_analysis_seconds = time.perf_counter() - initial_analysis_started
    previous_maximum_residual = None
    plateau_count = 0
    search_stall_count = 0
    search_directions: list[NativeMPS] = []

    targeted_roots = (
        config.retained_roots
        if config.targeted_roots is None
        else config.targeted_roots
    )
    
    has_passive_guards = config.active_roots < config.retained_roots

    for iteration in range(config.max_iterations + 1):
        cycle_started = time.perf_counter()
        (
            maximum_tensor_imaginary,
            maximum_relative_tensor_imaginary,
        ) = tensor_block_imaginary_diagnostics(states)
        row = {
            "iteration": iteration,
            "values": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in analysis["values"]
            ],
            "residual_norms": [
                float(value) for value in analysis["residual_norms"]
            ],
            "maximum_residual": analysis["maximum_residual"],
            "eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in analysis["eigenvalues"]
            ],
            "eigenvector_residual_norms": [
                float(value)
                for value in analysis["eigenvector_residual_norms"]
            ],
            "maximum_eigenvector_residual": analysis[
                "maximum_eigenvector_residual"
            ],
            "eigenpair_residual_order": analysis[
                "eigenpair_residual_order"
            ],
            "eigenpair_tracking": analysis["eigenpair_tracking"],
            "nearest_guard_residual": analysis["nearest_guard_residual"],
            "nearest_guard_index": analysis["nearest_guard_index"],
            "gram_frobenius_error": analysis["gram_error"],
            "maximum_bond_dimension": max(state.max_bond for state in states),
            "maximum_charge_block_bond": max(
                state.max_charge_block_bond for state in states
            ),
            "maximum_sector_summed_bond": max(
                state.max_total_bond for state in states
            ),
            "maximum_tensor_imaginary_element": maximum_tensor_imaginary,
            "maximum_relative_tensor_imaginary_norm": (
                maximum_relative_tensor_imaginary
            ),
            "correction_column_indices": analysis[
                "correction_column_indices"
            ],
            "search_stall_count_before_cycle": search_stall_count,
            "schur": analysis["schur"],
        }
        if previous_maximum_residual is not None:
            relative_change = abs(
                analysis["maximum_residual"] - previous_maximum_residual
            ) / max(previous_maximum_residual, 1.0e-300)
            plateau_count = (
                plateau_count + 1
                if relative_change <= config.plateau_relative_tolerance
                else 0
            )
            row["maximum_residual_relative_change"] = relative_change
        row["plateau_count"] = plateau_count
        if (
            analysis["maximum_residual"] <= config.residual_tolerance
            and (
                not has_passive_guards
                or analysis["nearest_guard_residual"]
                <= config.guard_residual_factor * config.residual_tolerance
            )
        ):
            converged = True
            row["accepted"] = True
            row["termination"] = (
                "active and nearest-guard contracted residual tolerances reached"
                if has_passive_guards
                else "targeted contracted residual tolerance reached"
            )
            row["cycle_seconds"] = time.perf_counter() - cycle_started
            history.append(row)
            if config.verbose:
                print(json.dumps({key: row[key] for key in (
                    "iteration", "values", "residual_norms",
                    "eigenvalues", "eigenvector_residual_norms",
                    "maximum_eigenvector_residual",
                    "maximum_residual", "nearest_guard_residual",
                    "termination", "cycle_seconds",
                )}), flush=True)
            break
        if (
            plateau_count >= config.plateau_iterations
            and analysis["maximum_residual"]
            <= config.plateau_residual_factor * config.residual_tolerance
            and (
                not has_passive_guards
                or analysis["nearest_guard_residual"]
                <= config.plateau_residual_factor * config.residual_tolerance
            )
        ):
            row["accepted"] = True
            row["termination"] = "bond-limited contracted residual plateau"
            row["cycle_seconds"] = time.perf_counter() - cycle_started
            history.append(row)
            if config.verbose:
                print(json.dumps({key: row[key] for key in (
                    "iteration", "values", "residual_norms",
                    "eigenvalues", "eigenvector_residual_norms",
                    "maximum_eigenvector_residual",
                    "maximum_residual", "nearest_guard_residual",
                    "plateau_count", "termination",
                    "cycle_seconds",
                )}), flush=True)
            break
        if iteration == config.max_iterations:
            row["accepted"] = False
            row["termination"] = "iteration budget exhausted"
            row["cycle_seconds"] = time.perf_counter() - cycle_started
            history.append(row)
            if config.verbose:
                print(json.dumps({key: row[key] for key in (
                    "iteration", "values", "residual_norms",
                    "eigenvalues", "eigenvector_residual_norms",
                    "maximum_eigenvector_residual",
                    "maximum_residual", "nearest_guard_residual",
                    "termination", "cycle_seconds",
                )}), flush=True)
            break

        phase_seconds = {}
        phase_started = time.perf_counter()
        residuals, residual_reports = _materialize_residuals(
            states,
            actions,
            operator,
            analysis,
            config,
            call_index=linear_call,
        )
        linear_call += 1
        phase_seconds["residual_src"] = time.perf_counter() - phase_started
        total_src_additions += len(residual_reports)
        corrections = []
        correction_rows = []
        projection_seconds = 0.0

        
        input_batch_started = time.perf_counter()
        input_call_base = linear_call
        input_basis_packed = pack_state_sites(states)

        def project_input(item):
            root, residual = item
            started = time.perf_counter()
            projected, report = _project_src(
                residual,
                states,
                config,
                call_index=input_call_base + root,
                basis_gram=analysis["gram"],
                basis_packed=input_basis_packed,
            )
            return projected, report, time.perf_counter() - started

        input_results = gpu_parallel_map(project_input, list(enumerate(residuals)))
        linear_call += len(residuals)
        phase_seconds["input_projection_src"] = (
            time.perf_counter() - input_batch_started
        )
        for root, (projected_residual, input_projection_report, _) in enumerate(
            input_results
        ):
            total_src_additions += len(
                input_projection_report["src_projections"]
            )
            if projected_residual is None:
                raise FloatingPointError(
                    f"projected GPLHR residual {root} vanished"
                )

        preconditioner_batch_started = time.perf_counter()
        preconditioner_call_base = linear_call

        def precondition(item):
            root, (projected_residual, _, _) = item
            started = time.perf_counter()
            output, report = _apply_preconditioner_src(
                projected_residual,
                operator,
                parameters,
                config,
                call_index=preconditioner_call_base + root,
                one_site_preconditioner=one_site_preconditioner,
            )
            return output, report, time.perf_counter() - started

        if multi_gpu_preconditioner is None:
            preconditioned_results = gpu_parallel_map(
                precondition, list(enumerate(input_results))
            )
        else:
            preconditioned_results = multi_gpu_preconditioner.apply_block(
                [row[0] for row in input_results],
                [preconditioner_call_base + root for root in range(len(input_results))],
            )
        linear_call += len(input_results)
        phase_seconds["preconditioner_src"] = (
            time.perf_counter() - preconditioner_batch_started
        )
        for _, preconditioner_report, _ in preconditioned_results:
            total_src_actions += preconditioner_report["mpo_actions"]
            total_src_additions += preconditioner_report[
                "src_linear_combinations"
            ]

        for root, (
            (preconditioned, preconditioner_report, local_preconditioner_seconds),
            (_, input_projection_report, input_projection_seconds),
            residual,
        ) in enumerate(zip(preconditioned_results, input_results, residuals)):
            projection_started = time.perf_counter()
            correction, projection_report = _project_src(
                preconditioned,
                [*states, *corrections],
                config,
                call_index=linear_call,
            )
            linear_call += 1
            total_src_additions += len(projection_report["src_projections"])
            if correction is None:
                correction, projection_report = _project_src(
                    residual,
                    [*states, *corrections],
                    config,
                    call_index=linear_call,
                )
                linear_call += 1
                total_src_additions += len(
                    projection_report["src_projections"]
                )
            local_projection_seconds = time.perf_counter() - projection_started
            projection_seconds += local_projection_seconds
            if correction is None:
                raise FloatingPointError(f"GPLHR correction {root} vanished")
            correction.validate(config.max_bond)
            corrections.append(correction)
            correction_rows.append(
                {
                    "correction": root,
                    "schur_column": analysis[
                        "correction_column_indices"
                    ][root],
                    "residual_src": residual_reports[root],
                    "input_projection": input_projection_report,
                    "input_projection_seconds": input_projection_seconds,
                    "preconditioner": preconditioner_report,
                    "projection": projection_report,
                    "preconditioner_seconds": local_preconditioner_seconds,
                    "projection_seconds": local_projection_seconds,
                }
            )
        phase_seconds["projection_src"] = projection_seconds

        phase_started = time.perf_counter()
        projected_search_directions = []
        search_projection_reports = []
        if config.reproject_search_directions:
            for column, direction in enumerate(search_directions):
                projected_direction, projection_report = _project_src(
                    direction,
                    [*states, *corrections, *projected_search_directions],
                    config,
                    call_index=linear_call,
                )
                linear_call += 1
                total_src_additions += len(
                    projection_report["src_projections"]
                )
                search_projection_reports.append(
                    {"column": column, **projection_report}
                )
                if projected_direction is not None:
                    projected_search_directions.append(projected_direction)
        else:
            projected_search_directions = list(search_directions)
            search_projection_reports = [
                {
                    "column": column,
                    "bypassed": True,
                    "reason": (
                        "the complete [V,W,P] metric is whitened with a rank "
                        "guard in harmonic restart"
                    ),
                    "src_projections": [],
                }
                for column in range(len(search_directions))
            ]
        if len(projected_search_directions) > len(search_directions):
            raise AssertionError("projected GPLHR P block grew unexpectedly")
        phase_seconds["search_direction_projection_src"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        correction_actions, correction_action_reports, action_seconds = (
            _apply_block_src(
                operator, corrections, config, call_index=action_call
            )
        )
        phase_seconds["correction_h_src"] = time.perf_counter() - phase_started
        action_call += 1
        total_src_actions += len(correction_actions)

        phase_started = time.perf_counter()
        (
            projected_search_actions,
            search_action_reports,
            search_action_seconds,
        ) = _apply_block_src(
            operator,
            projected_search_directions,
            config,
            call_index=action_call,
        )
        if projected_search_directions:
            action_call += 1
        total_src_actions += len(projected_search_actions)
        phase_seconds["search_direction_h_src"] = (
            time.perf_counter() - phase_started
        )

        
        adaptive_s1 = []
        adaptive_s1_actions = []
        adaptive_report = {
            "used": False,
            "trigger_search_stall_count": search_stall_count,
        }
        if config.adaptive_m1 and search_stall_count >= 2:
            adaptive_started = time.perf_counter()
            arnoldi_sources, arnoldi_reports = (
                _materialize_arnoldi_residuals(
                    corrections,
                    operator,
                    analysis["factor_a"][
                        : len(corrections), : len(corrections)
                    ],
                    analysis["factor_b"][
                        : len(corrections), : len(corrections)
                    ],
                    config,
                    call_index=linear_call,
                )
            )
            linear_call += 1
            total_src_additions += len(arnoldi_sources)
            source_projection_reports = []
            preconditioner_reports = []
            output_projection_reports = []
            for source in arnoldi_sources:
                projected_source, source_projection = _project_src(
                    source,
                    states,
                    config,
                    call_index=linear_call,
                )
                linear_call += 1
                total_src_additions += len(
                    source_projection["src_projections"]
                )
                if projected_source is None:
                    continue
                preconditioned, preconditioner = _apply_preconditioner_src(
                    projected_source,
                    operator,
                    parameters,
                    config,
                    call_index=linear_call,
                    one_site_preconditioner=one_site_preconditioner,
                )
                linear_call += 1
                total_src_actions += preconditioner["mpo_actions"]
                total_src_additions += preconditioner[
                    "src_linear_combinations"
                ]
                candidate, output_projection = _project_src(
                    preconditioned,
                    [*states, *corrections, *adaptive_s1],
                    config,
                    call_index=linear_call,
                )
                linear_call += 1
                total_src_additions += len(
                    output_projection["src_projections"]
                )
                source_projection_reports.append(source_projection)
                preconditioner_reports.append(preconditioner)
                output_projection_reports.append(output_projection)
                if candidate is not None:
                    adaptive_s1.append(candidate)
            if adaptive_s1:
                (
                    adaptive_s1_actions,
                    adaptive_action_reports,
                    adaptive_action_seconds,
                ) = _apply_block_src(
                    operator,
                    adaptive_s1,
                    config,
                    call_index=action_call,
                )
                action_call += 1
                total_src_actions += len(adaptive_s1_actions)
            else:
                adaptive_action_reports = []
                adaptive_action_seconds = 0.0
            adaptive_report = {
                "used": bool(adaptive_s1),
                "trigger_search_stall_count": search_stall_count,
                "columns": len(adaptive_s1),
                "arnoldi_residuals": arnoldi_reports,
                "source_projections": source_projection_reports,
                "preconditioners": preconditioner_reports,
                "output_projections": output_projection_reports,
                "h_actions": adaptive_action_reports,
                "seconds": time.perf_counter() - adaptive_started,
            }
            phase_seconds["adaptive_m1"] = adaptive_report["seconds"]

        trial_states = [
            *states,
            *corrections,
            *adaptive_s1,
            *projected_search_directions,
        ]
        trial_actions = [
            *actions,
            *correction_actions,
            *adaptive_s1_actions,
            *projected_search_actions,
        ]
        phase_started = time.perf_counter()
        (
            restart_coefficients,
            search_restart_coefficients,
            candidate_factor_a,
            candidate_factor_b,
            harmonic_report,
        ) = _harmonic_restart(
            trial_states, operator, config, actions=trial_actions
        )
        phase_seconds["harmonic_contractions_qz"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        all_restart_coefficients = np.concatenate(
            [restart_coefficients, search_restart_coefficients], axis=1
        )
        restart_states, restart_reports = rotate_src(
            trial_states,
            all_restart_coefficients,
            config.max_bond,
            seed=config.seed + 104729 * linear_call,
            power_steps=config.src_power_steps,
        )
        candidate_states = restart_states[: config.retained_roots]
        candidate_search_directions = restart_states[config.retained_roots :]
        linear_call += 1
        phase_seconds["restart_rotation_src"] = (
            time.perf_counter() - phase_started
        )
        total_src_additions += len(restart_reports)
        phase_started = time.perf_counter()
        candidate_gram = gram_matrix(candidate_states)
        candidate_metric = metric_whitener(
            candidate_gram, config.gram_cutoff
        )
        if candidate_metric.retained != config.retained_roots:
            raise FloatingPointError("SRC restart lost retained block rank")
        candidate_metric_report = {
            "method": (
                "actual compressed-MPS metric guard; logical whitening is "
                "kept in coefficient space"
            ),
            "retained_rank": candidate_metric.retained,
            "condition_number": candidate_metric.condition_number,
            "gram_frobenius_error": float(
                np.linalg.norm(candidate_gram - np.eye(config.retained_roots))
            ),
            "additional_lossy_mps_orthogonalization": False,
        }
        phase_seconds["restart_metric_contractions"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        candidate_actions, restart_action_reports, restart_action_seconds = (
            _apply_block_src(
                operator, candidate_states, config, call_index=action_call
            )
        )
        phase_seconds["restart_h_src"] = time.perf_counter() - phase_started
        action_call += 1
        total_src_actions += len(candidate_actions)
        phase_started = time.perf_counter()
        candidate_analysis = _qfree_srr_analysis(
            candidate_states,
            operator,
            config,
            candidate_factor_a,
            candidate_factor_b,
            actions=candidate_actions,
        )
        candidate_analysis = _align_eigenpair_diagnostics(
            states,
            analysis,
            candidate_states,
            candidate_analysis,
        )
        phase_seconds["candidate_analysis_contractions"] = (
            time.perf_counter() - phase_started
        )

        
        accepted = True
        predicted_residual = harmonic_report["predicted_maximum_residual"]
        rho_search = predicted_residual / max(
            row["maximum_residual"], 1.0e-300
        )
        rho_trunc = candidate_analysis["maximum_residual"] / max(
            predicted_residual, np.finfo(np.float64).eps
        )
        search_stall_count = search_stall_count + 1 if rho_search > 0.8 else 0
        states = candidate_states
        actions = candidate_actions
        analysis = candidate_analysis
        search_directions = candidate_search_directions
        row.update(
            residual_materialization=residual_reports,
            corrections=correction_rows,
            correction_h_actions=correction_action_reports,
            search_direction_projection=search_projection_reports,
            adaptive_m1=adaptive_report,
            harmonic=harmonic_report,
            restart_src=restart_reports,
            restart_metric_guard=candidate_metric_report,
            restart_h_actions=restart_action_reports,
            phase_seconds=phase_seconds,
            candidate_maximum_residual=candidate_analysis["maximum_residual"],
            rho_search=float(rho_search),
            rho_trunc=float(rho_trunc),
            residual_ratio=(
                candidate_analysis["maximum_residual"]
                / max(row["maximum_residual"], 1.0e-300)
            ),
            accepted=accepted,
            restart_acceptance_policy="unconditional standard GPLHR restart",
            h_action_seconds=action_seconds + restart_action_seconds,
            cycle_seconds=time.perf_counter() - cycle_started,
        )
        history.append(row)
        if config.verbose:
            print(json.dumps({key: row[key] for key in (
                "iteration", "values", "residual_norms",
                "eigenvalues", "eigenvector_residual_norms",
                "maximum_eigenvector_residual", "maximum_residual",
                "nearest_guard_residual", "nearest_guard_index",
                "candidate_maximum_residual", "rho_search", "rho_trunc",
                "residual_ratio", "accepted", "cycle_seconds",
            )}), flush=True)
        previous_maximum_residual = row["maximum_residual"]

    final_analysis = analysis
    active_triangular = final_analysis["triangular"][
        : config.active_roots, : config.active_roots
    ]
    raw_values, raw_eigenvectors = scipy.linalg.eig(
        active_triangular, check_finite=False
    )
    raw_coefficients = (
        final_analysis["transform"][:, : config.active_roots]
        @ raw_eigenvectors
    )
    raw_coefficient_gram = (
        raw_coefficients.conj().T
        @ final_analysis["gram"]
        @ raw_coefficients
    )
    raw_norms = np.sqrt(np.maximum(np.real(np.diag(raw_coefficient_gram)), 0.0))
    if float(np.min(raw_norms)) <= 1.0e-14:
        raise FloatingPointError("final Ritz vector has negligible MPS norm")
    normalized_raw_coefficients = raw_coefficients / raw_norms[None, :]
    final_order, final_matched_overlaps = _maximum_overlap_eigenpair_order(
        final_analysis["eigenvector_coefficients"],
        final_analysis["gram"],
        normalized_raw_coefficients,
    )
    values = raw_values[final_order]
    coefficients = raw_coefficients[:, final_order]
    final_states, final_rotation_reports = rotate_src(
        states,
        coefficients,
        config.max_bond,
        seed=config.seed + 104729 * linear_call,
        power_steps=config.src_power_steps,
    )
    linear_call += 1
    total_src_additions += len(final_rotation_reports)
    final_states = [normalize(state) for state in final_states]
    final_actions, final_action_reports, final_action_seconds = _apply_block_src(
        operator, final_states, config, call_index=action_call
    )
    total_src_actions += len(final_actions)
    final_values = np.asarray(
        [
            overlap(state, action) / overlap(state, state)
            for state, action in zip(final_states, final_actions)
        ]
    )
    if config.final_residual_from_compressed_actions:
        final_gram, final_projected, final_action_gram = (
            _block_matrices_from_actions(final_states, final_actions)
        )
    else:
        final_gram, final_projected, final_action_gram = _block_matrices(
            final_states, operator
        )
    final_residual_gram = _residual_gram(
        final_gram,
        final_projected,
        final_action_gram,
        np.eye(config.active_roots, dtype=np.complex128),
        np.diag(final_values),
    )
    final_contracted_residuals = _residual_norms(final_residual_gram)
    materialized_residual_norms = []
    final_residual_reports = []
    for state, action, value in zip(final_states, final_actions, final_values):
        residual, report = src_sum(
            [action, state],
            [1.0, -value],
            config.max_bond,
            seed=config.seed + 104729 * linear_call,
            power_steps=config.src_power_steps,
        )
        linear_call += 1
        materialized_residual_norms.append(norm(residual))
        final_residual_reports.append(report)
        total_src_additions += 1

    
    action_scales = np.sqrt(
        np.maximum(np.real(np.diag(final_action_gram)), 0.0)
    )
    roundoff_thresholds = 64.0 * np.sqrt(
        np.finfo(np.float64).eps
    ) * np.maximum(action_scales, 1.0)
    reported_residuals = np.where(
        final_contracted_residuals <= roundoff_thresholds,
        np.asarray(materialized_residual_norms),
        final_contracted_residuals,
    )
    maximum_final_eigenvalue_imaginary_part = float(
        np.max(np.abs(np.imag(final_values)))
    )
    real_spectrum_pass = bool(
        maximum_final_eigenvalue_imaginary_part
        <= config.real_eigenvalue_tolerance
    )
    maximum_history_eigenvalue_imaginary_part = max(
        (
            abs(float(value["imag"]))
            for row in history
            for value in row.get("eigenvalues", [])
        ),
        default=0.0,
    )
    if config.require_real_ritz and not real_spectrum_pass:
        raise FloatingPointError(
            "real-Ritz contract failed after final MPS rotation: "
            f"maximum |Im(E)|={maximum_final_eigenvalue_imaginary_part:.6e} "
            f"exceeds {config.real_eigenvalue_tolerance:.6e}"
        )
    (
        maximum_final_tensor_imaginary,
        maximum_final_relative_tensor_imaginary,
    ) = tensor_block_imaginary_diagnostics(final_states)

    multi_gpu_preconditioner_audit = (
        multi_gpu_preconditioner.audit()
        if multi_gpu_preconditioner is not None
        else {
            "enabled": False,
            "devices": 1,
            "full_hilbert_vectors": 0,
            "fixed_sector_vectors": 0,
        }
    )
    if multi_gpu_preconditioner is not None:
        multi_gpu_preconditioner.close()

    diagnostics = {
        "algorithm": "standard native symmetry-block MPS-GPLHR(0)",
        "standard_gplhr0_structure": {
            "initial_trial_block": "[V_retained,W_targeted]",
            "subsequent_trial_block": "[V_retained,W_targeted,P_targeted]",
            "restart": (
                "b-column harmonic Schur block; P is the orthogonalized "
                "non-V-row recurrence from the same selected block"
            ),
            "restart_acceptance": "unconditional after rank guards",
            "residual_projection_before_preconditioner": True,
            "correction_projection_after_preconditioner": True,
            "adaptive_m1_enabled": config.adaptive_m1,
            "harmonic_extraction": (
                "Q-free SRR factors MA, MB from harmonic QZ; residual is "
                "A V MB - V MA; no post-QZ ordinary Rayleigh-Ritz extraction"
            ),
        },
        "compression_policy": {
            "hamiltonian_action": "SRC",
            "mps_addition_rotation_residual_restart": "symmetry-resolved SRC",
            "hidden_normalization_in_compressors": False,
            "state_processing": (
                "each output vector is compressed as one independent MPS; "
                "no fused block-MPS and no shared bond budget"
            ),
        },
        "real_structure": {
            "enabled": config.preserve_real_structure,
            "operator_maximum_imaginary_entry": float(
                maximum_operator_imaginary_entry
            ),
            "initializer": "real symmetry-native random MPS",
            "src_sketch": "real whenever its input is numerically real",
            "small_dense_factorization": (
                "real Schur/QZ with quasi-triangular 2x2 blocks whenever "
                "projected data are numerically real"
            ),
            "maximum_final_tensor_imaginary_element": (
                maximum_final_tensor_imaginary
            ),
            "maximum_final_relative_tensor_imaginary_norm": (
                maximum_final_relative_tensor_imaginary
            ),
            "maximum_final_eigenvalue_imaginary_part": (
                maximum_final_eigenvalue_imaginary_part
            ),
            "maximum_history_eigenvalue_imaginary_part": (
                maximum_history_eigenvalue_imaginary_part
            ),
            "eigenvalue_imaginary_tolerance": (
                config.real_eigenvalue_tolerance
            ),
            "real_spectrum_pass": real_spectrum_pass,
        },
        "config": asdict(config),
        "multi_gpu_preconditioner": multi_gpu_preconditioner_audit,
        "initial_orthogonalization": initial_orthogonalization,
        "initial_subspace_filter": {
            "operator": "initial_filter_upper_bound * I - H",
            "steps": config.initial_filter_steps,
            "upper_bound": config.initial_filter_upper_bound,
            "h_action_seconds": initial_filter_action_seconds,
            "reports": initial_filter_reports,
        },
        "initial_h_actions": initial_action_reports,
        "initial_h_action_seconds": initial_action_seconds,
        "initial_harmonic_srr": initial_harmonic_report,
        "initial_harmonic_rotation_src": initial_harmonic_rotation_reports,
        "initial_srr_h_actions": initial_srr_action_reports,
        "initial_srr_h_action_seconds": initial_srr_action_seconds,
        "initial_analysis_seconds": initial_analysis_seconds,
        "final_rotation_src": final_rotation_reports,
        "final_h_actions": final_action_reports,
        "final_h_action_seconds": final_action_seconds,
        "final_residual_src": final_residual_reports,
        "final_contracted_residual_norms": [
            float(value) for value in final_contracted_residuals
        ],
        "final_src_materialized_residual_norms": [
            float(value) for value in materialized_residual_norms
        ],
        "final_reported_residual_norms": [
            float(value) for value in reported_residuals
        ],
        "final_residual_contraction": (
            "Gram of freshly materialized bond-capped H|psi> actions"
            if config.final_residual_from_compressed_actions
            else "direct MPO double layer <psi|H^dagger H|psi>"
        ),
        "final_eigenpair_tracking": {
            "method": (
                "maximum absolute physical MPS overlap with the last "
                "iteration's persistent root identities"
            ),
            "raw_final_indices_by_persistent_root": [
                int(index) for index in final_order
            ],
            "matched_absolute_overlaps": [
                float(value) for value in final_matched_overlaps
            ],
            "minimum_matched_absolute_overlap": float(
                np.min(final_matched_overlaps)
            ),
        },
        "final_residual_roundoff_fallback_used": [
            bool(value)
            for value in final_contracted_residuals <= roundoff_thresholds
        ],
        "src_hamiltonian_actions": total_src_actions,
        "src_linear_combination_operations": total_src_additions,
        "fixed_sector_vectors_in_outer_iteration": 0,
        "full_hilbert_vectors_in_outer_iteration": 0,
        "maximum_bond_dimension": max(state.max_bond for state in final_states),
        "maximum_charge_block_bond": max(
            state.max_charge_block_bond for state in final_states
        ),
        "maximum_sector_summed_bond": max(
            state.max_total_bond for state in final_states
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return NativeGPLHRResult(
        states=final_states,
        actions=final_actions,
        values=final_values,
        residual_norms=np.asarray(reported_residuals),
        converged=converged,
        iterations=int(history[-1]["iteration"]),
        history=history,
        diagnostics=diagnostics,
    )


__all__ = [
    "NativeGPLHRConfig",
    "NativeGPLHRResult",
    "initial_hubbard_block",
    "solve_native_gplhr",
]
