from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import scipy.sparse.linalg

from renormalizer import Mpo, Mps
from renormalizer.mps.block_env import BlockEnvData, block_hop_expr_two_site, dense_to_block_env
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.lib import Environ, cvec2cmat
from renormalizer.mps.matrix import asnumpy, asxp, tensordot
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.utils import CompressConfig, CompressCriteria


@dataclass
class ShiftedSolveReport:
    """Diagnostics for :func:`solve_shifted_two_site`."""

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
    solver: str = "two-site-mals"
    sweep_seconds: list[float] = field(default_factory=list)
    sweep_local_iterations: list[int] = field(default_factory=list)
    stop_reason: str = "sweep_limit"
    best_sweep: int = 0


class _ConjugatedSites:
    """Read the current bra tensor without copying the entire MPS per site."""

    def __init__(self, state):
        self.state = state

    def __getitem__(self, site):
        return self.state[site].conj()


def _absorb_coefficient(state: Mps) -> Mps:
    """Return a copy whose physical amplitude lives only in site tensors."""

    result = state.copy()
    coefficient = complex(result.coeff)
    
    if coefficient != 1.0:
        result.scale(coefficient, inplace=True)
    result.coeff = 1.0 + 0.0j if result.is_complex else 1.0
    return result


def _inner(left: Mps, right: Mps) -> complex:
    """Full MPS inner product, including Renormalizer's scalar coefficient."""

    return (
        complex(left.coeff).conjugate()
        * complex(right.coeff)
        * left.conj().dot(right)
    )


def _norm(state: Mps) -> float:
    norm_squared = _inner(state, state)
    scale = abs(norm_squared)
    if abs(norm_squared.imag) > 1.0e-10 * max(scale, 1.0):
        raise FloatingPointError(
            f"MPS norm has a material imaginary part: {norm_squared}"
        )
    if norm_squared.real < -1.0e-10 * max(scale, 1.0):
        raise FloatingPointError(f"negative MPS norm squared: {norm_squared}")
    return float(max(norm_squared.real, 0.0) ** 0.5)


def shifted_mpo(hamiltonian: Mpo, shift: complex) -> Mpo:
   

    identity = Mpo.identity(hamiltonian.model)
    return identity.scale(complex(shift)).add(hamiltonian.scale(-1.0))


def _global_shifted_residual_direct_sum(
    hamiltonian: Mpo,
    rhs: Mps,
    solution: Mps,
    shift: complex,
) -> float:


    rhs_work = _absorb_coefficient(rhs)
    solution_work = _absorb_coefficient(solution)
    solution_work.ensure_right_canonical()
    image = shifted_mpo(hamiltonian, shift).apply(solution_work)
    rhs_work.ensure_right_canonical()
    image.ensure_right_canonical()
    rhs_norm = _norm(rhs_work)
    if rhs_norm == 0.0:
        raise ValueError("the shifted-system right-hand side is zero")

    residual_state = rhs_work.add(image.scale(-1.0))
    residual_state.ensure_right_canonical()
    return _norm(residual_state) / rhs_norm


def global_shifted_residual(
    hamiltonian: Mpo,
    rhs: Mps,
    solution: Mps,
    shift: complex,
) -> float:
    

    rhs_work = _absorb_coefficient(rhs)
    solution_work = _absorb_coefficient(solution)
    rhs_squared = complex(_inner(rhs_work, rhs_work))
    if rhs_squared.real <= 0.0:
        raise ValueError("the shifted-system right-hand side is zero")
    if abs(rhs_squared.imag) > 1.0e-11 * max(abs(rhs_squared), 1.0):
        return _global_shifted_residual_direct_sum(
            hamiltonian, rhs_work, solution_work, shift
        )

    operator = shifted_mpo(hamiltonian, shift)
    
    cross = complex(
        solution_work.expectation(operator, self_conj=rhs_work.conj())
    )
    squared_operator = operator.conj_trans().apply(operator)
    image_squared = complex(solution_work.expectation(squared_operator))
    residual_squared_complex = rhs_squared + image_squared - cross - cross.conjugate()
    component_scale = (
        abs(rhs_squared) + abs(image_squared) + 2.0 * abs(cross)
    )
    residual_squared = float(residual_squared_complex.real)

   
    cancellation_guard = 1.0e-10 * max(component_scale, 1.0e-300)
    imaginary_guard = 1.0e-11 * max(component_scale, 1.0)
    if (
        np.isfinite(residual_squared)
        and residual_squared > cancellation_guard
        and abs(residual_squared_complex.imag) <= imaginary_guard
    ):
        return float(np.sqrt(residual_squared / rhs_squared.real))
    return _global_shifted_residual_direct_sum(
        hamiltonian, rhs_work, solution_work, shift
    )


def _project_rhs_center(
    rhs: Mps,
    identity: Mpo,
    cross_environ: Environ,
    solution: Mps,
    center: list[int],
    left_index: int,
    right_index: int,
    left_method: str,
    right_method: str,
):
    
    solution_conjugate = _ConjugatedSites(solution)
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
    rhs_center = tensordot(rhs[center[0]], rhs[center[1]], axes=(-1, 0))
    expression = hop_expr(
        left,
        right,
        [asxp(identity[index]) for index in center],
        rhs_center.shape,
    )
    return expression(rhs_center)


def _half_sweep(
    solution: Mps,
    shifted: Mpo,
    identity: Mpo,
    rhs: Mps,
    *,
    local_restart: int,
    local_maxiter: int,
    local_tolerance: float,
    use_block_env: bool,
    operator_environ=None,
    cross_environ=None,
    block_boundary_env=False,
):
   

    environment_domain = "R" if solution.to_right else "L"
    if operator_environ is None:
        operator_environ = Environ(
            solution, shifted, environment_domain,
            use_block_env=use_block_env, block_env_min_bond_dim=0,
        )
    if cross_environ is None:
        cross_environ = Environ(
            rhs, identity, environment_domain,
            mps_conj=_ConjugatedSites(solution),
        )

    local_solves = 0
    local_iterations = 0
    local_failures = 0
    block_local_solves = 0
    max_active_dimension = 0
    max_local_residual = 0.0

    for site in solution.iter_idx_list(full=True):
        if (solution.to_right and site == solution.site_num - 1) or (
            (not solution.to_right) and site == 0
        ):
            break

        if solution.to_right:
            left_method, right_method = "System", "Enviro"
            left_index = site - 1
            center = [site, site + 1]
            right_index = site + 2
        else:
            left_method, right_method = "Enviro", "System"
            left_index = site - 2
            center = [site - 1, site]
            right_index = site + 1

        qn_big_left, qn_big_right, qn_matrix = solution._get_big_qn(center)
        qn_mask = get_qn_mask(qn_matrix, solution.qntot)
        active_dimension = int(np.sum(asnumpy(qn_mask)))
        if active_dimension == 0:
            raise RuntimeError(f"empty local quantum-number sector at {center}")
        max_active_dimension = max(max_active_dimension, active_dimension)

        left = operator_environ.GetLR(
            "L",
            left_index,
            solution,
            shifted,
            method=left_method,
            raw=use_block_env,
        )
        right = operator_environ.GetLR(
            "R",
            right_index,
            solution,
            shifted,
            method=right_method,
            raw=use_block_env,
        )
        local_shape = tuple(qn_mask.shape)
        local_mpo = [asxp(shifted[index]) for index in center]

        if use_block_env and block_boundary_env:
           
            if not isinstance(left, BlockEnvData):
                index = center[0]
                left = dense_to_block_env(left, solution.qn[index],
                    shifted.qn[index], solution.qn[index], operator_environ.block_env_threshold)
            if not isinstance(right, BlockEnvData):
                index = center[1]+1
                right = dense_to_block_env(right, solution.qn[index],
                    shifted.qn[index], solution.qn[index], operator_environ.block_env_threshold)

        if isinstance(left, BlockEnvData) and isinstance(right, BlockEnvData):
            block_env_cache = getattr(operator_environ, "_block_env_cache", None)
            expression = block_hop_expr_two_site(
                left,
                right,
                local_mpo[0],
                local_mpo[1],
                local_shape,
                qn_mask,
                solution.qn[center[0]],
                solution.qn[center[1]],
                solution.qn[center[1] + 1],
                shifted.qn[center[0]],
                shifted.qn[center[1]],
                shifted.qn[center[1] + 1],
                operator_environ.block_env_threshold,
                None
                if block_env_cache is None
                else block_env_cache.symbolic_nonzero_keys(center[0]),
                None
                if block_env_cache is None
                else block_env_cache.symbolic_nonzero_keys(center[1]),
                packed_only=True,
                packed_matrix_limit_bytes=8 * 1024 * 1024,
                op0_blocks=None if block_env_cache is None else block_env_cache.mpo_lr_blocks(center[0]),
                op1_blocks=None if block_env_cache is None else block_env_cache.mpo_lr_blocks(center[1]),
            )
            if expression.center_layout.size != active_dimension:
                raise RuntimeError(
                    "block center layout does not match the active QN sector"
                )

            def apply(vector):
                return np.asarray(expression.apply_packed(np.asarray(vector)))

            block_local_solves += 1
        else:
            
            if isinstance(left, BlockEnvData):
                left = left.to_dense()
            if isinstance(right, BlockEnvData):
                right = right.to_dense()
            expression = hop_expr(
                asxp(left),
                asxp(right),
                local_mpo,
                local_shape,
            )

            def apply(vector):
                structured = asxp(cvec2cmat(np.asarray(vector), qn_mask))
                image = expression(structured)
                return np.asarray(asnumpy(image))[qn_mask]

        projected_rhs = _project_rhs_center(
            rhs,
            identity,
            cross_environ,
            solution,
            center,
            left_index,
            right_index,
            left_method,
            right_method,
        )
        projected_rhs = np.asarray(asnumpy(projected_rhs))[qn_mask]
        current_center = tensordot(
            solution[center[0]], solution[center[1]], axes=(-1, 0)
        )
        initial = np.asarray(asnumpy(current_center))[qn_mask]

        operator = scipy.sparse.linalg.LinearOperator(
            (active_dimension, active_dimension),
            matvec=apply,
            dtype=np.result_type(initial.dtype, projected_rhs.dtype, complex),
        )
        iteration_residuals: list[float] = []
        solved, info = scipy.sparse.linalg.gmres(
            operator,
            projected_rhs,
            x0=initial,
            rtol=float(local_tolerance),
            atol=0.0,
            restart=min(int(local_restart), active_dimension),
            maxiter=int(local_maxiter),
            callback=lambda value: iteration_residuals.append(float(value)),
            callback_type="pr_norm",
        )
        true_local = np.linalg.norm(apply(solved) - projected_rhs)
        rhs_norm = np.linalg.norm(projected_rhs)
        relative_local = float(true_local / max(rhs_norm, 1.0e-300))

        local_solves += 1
        local_iterations += len(iteration_residuals)
        local_failures += int(info != 0)
        max_local_residual = max(max_local_residual, relative_local)

        structured_solution = asxp(cvec2cmat(solved, qn_mask))
        solution._update_mps(
            structured_solution,
            center,
            qn_big_left,
            qn_big_right,
            percent=0.0,
        )

    solution._switch_direction()
    return (
        local_solves,
        local_iterations,
        local_failures,
        block_local_solves,
        max_active_dimension,
        max_local_residual,
    )


def solve_shifted_two_site(
    hamiltonian: Mpo,
    rhs: Mps,
    shift: complex,
    x0: Mps | None = None,
    *,
    max_sweeps: int = 8,
    local_restart: int = 64,
    local_maxiter: int = 4,
    local_tolerance: float = 1.0e-10,
    bond_cap: int = 64,
    cutoff: float = 1.0e-12,
    global_tolerance: float = 1.0e-8,
    use_block_env: bool = False,
    stagnation_sweeps: int = 2,
    stagnation_tolerance: float = 1.0e-3,
    residual_evaluator=None,
    block_boundary_env=False,
) -> tuple[Mps, ShiftedSolveReport]:
    
    operator_physical_dims = tuple(int(value) for value in hamiltonian.pbond_list)
    rhs_physical_dims = tuple(int(value) for value in rhs.pbond_list)
    if (
        hamiltonian.site_num != rhs.site_num
        or operator_physical_dims != rhs_physical_dims
    ):
        raise ValueError("Hamiltonian and right-hand side use incompatible models")
    if x0 is not None and (
        x0.site_num != rhs.site_num
        or tuple(int(value) for value in x0.pbond_list) != rhs_physical_dims
    ):
        raise ValueError("initial guess and right-hand side use incompatible models")
    if max_sweeps < 1 or local_restart < 1 or local_maxiter < 1:
        raise ValueError("sweep and GMRES iteration counts must be positive")
    if bond_cap < 1:
        raise ValueError("bond_cap must be positive")
    if not (0.0 < cutoff < 1.0):
        raise ValueError("cutoff must lie strictly between zero and one")
    if local_tolerance <= 0.0 or global_tolerance <= 0.0:
        raise ValueError("residual tolerances must be positive")
    if stagnation_sweeps < 0 or not (0.0 <= stagnation_tolerance < 1.0):
        raise ValueError("invalid stagnation controls")

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

    solution.compress_config = CompressConfig(
        CompressCriteria.both,
        threshold=float(cutoff),
        max_bonddim=int(bond_cap),
    )
    if max(solution.bond_dims) > bond_cap:
        solution.canonicalise().compress()
    solution.ensure_right_canonical()
    shifted = shifted_mpo(hamiltonian, shift)
    identity = Mpo.identity(hamiltonian.model)
    
    operator_environ = Environ(
        solution, shifted, "R", use_block_env=bool(use_block_env),
        block_env_min_bond_dim=0,
    )
    cross_environ = Environ(
        rhs_work, identity, "R", mps_conj=_ConjugatedSites(solution),
    )

    total_local_solves = 0
    total_local_iterations = 0
    total_local_failures = 0
    total_block_local_solves = 0
    max_active_dimension = 0
    max_local_residual = 0.0
    residual_history: list[float] = []
    completed_sweeps = 0
    sweep_seconds = []
    sweep_iterations = []
    best_residual = float("inf")
    best_solution = None
    best_sweep = 0
    stagnant = 0
    stop_reason = "sweep_limit"

    for sweep in range(1, max_sweeps + 1):
        sweep_started = time.perf_counter()
        previous_iterations = total_local_iterations
        for _ in range(2):
            statistics = _half_sweep(
                solution,
                shifted,
                identity,
                rhs_work,
                local_restart=local_restart,
                local_maxiter=local_maxiter,
                local_tolerance=local_tolerance,
                use_block_env=bool(use_block_env),
                operator_environ=operator_environ,
                cross_environ=cross_environ,
                block_boundary_env=block_boundary_env,
            )
            (
                local_solves,
                iterations,
                failures,
                block_solves,
                active,
                local_residual,
            ) = statistics
            total_local_solves += local_solves
            total_local_iterations += iterations
            total_local_failures += failures
            total_block_local_solves += block_solves
            max_active_dimension = max(max_active_dimension, active)
            max_local_residual = max(max_local_residual, local_residual)

        completed_sweeps = sweep
        residual = (residual_evaluator or global_shifted_residual)(
            hamiltonian, rhs_work, solution, shift
        )
        residual_history.append(residual)
        sweep_seconds.append(time.perf_counter() - sweep_started)
        sweep_iterations.append(total_local_iterations - previous_iterations)
        if residual < (1.0 - stagnation_tolerance) * best_residual:
            stagnant = 0
        else:
            stagnant += 1
        if residual < best_residual:
            best_residual = residual
            best_solution = solution.copy()
            best_sweep = sweep
        if residual <= global_tolerance:
            stop_reason = "converged"
            break
        if stagnation_sweeps and stagnant >= stagnation_sweeps:
            stop_reason = "stagnation"
            break

    if best_solution is None:
        raise FloatingPointError("no finite shifted-system residual")
    solution = best_solution
    final_residual = best_residual
    maximum_solution_bond = max(int(value) for value in solution.bond_dims)
    if maximum_solution_bond > int(bond_cap):
        raise RuntimeError(
            f"shifted solver retained bond {maximum_solution_bond} exceeds "
            f"requested cap {bond_cap}"
        )
    report = ShiftedSolveReport(
        shift=complex(shift),
        sweeps=completed_sweeps,
        local_solves=total_local_solves,
        local_iterations=total_local_iterations,
        local_failures=total_local_failures,
        max_active_dimension=max_active_dimension,
        max_local_relative_residual=max_local_residual,
        global_relative_residual=final_residual,
        global_residual_history=residual_history,
        converged=final_residual <= global_tolerance,
        max_bond_dimension=maximum_solution_bond,
        use_block_env=bool(use_block_env),
        block_local_solves=total_block_local_solves,
        seconds=time.perf_counter() - started,
        sweep_seconds=sweep_seconds,
        sweep_local_iterations=sweep_iterations,
        stop_reason=stop_reason,
        best_sweep=best_sweep,
    )
    return solution, report
