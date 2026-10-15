from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


VectorAction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BiCGSTABResult:
   
    solution: np.ndarray
    relative_residual: float
    iterations: int
    matvecs: int
    converged: bool
    breakdowns: int


def _checked_vector(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    source: str,
) -> np.ndarray:
    """Return an owned, finite vector supplied by a user callback."""

    vector = np.asarray(value)
    if vector.shape != shape:
        raise ValueError(
            f"{source} returned shape {vector.shape}; expected {shape}"
        )
    vector = np.array(vector, dtype=dtype, copy=True)
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError(f"{source} returned a non-finite vector")
    return vector


def _near_breakdown(value: complex, scale: float, tolerance: float) -> bool:
    """Scale-invariant test for a BiCGSTAB denominator breakdown."""

    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        return True
    if scale == 0.0 or not np.isfinite(scale):
        return True
    return bool(abs(value) <= tolerance * scale)


def _restart_shadow(residual: np.ndarray, breakdowns: int) -> np.ndarray:
    

    shadow = residual.copy()
    if breakdowns == 0 or residual.size < 2:
        return shadow

    residual_norm = float(np.linalg.norm(residual))
    largest = int(np.argmax(np.abs(residual)))
    coordinate = (largest + breakdowns) % residual.size
    phase = np.exp(0.5j * np.pi * breakdowns)
    shadow[coordinate] += 0.1 * phase * residual_norm
    return shadow


def reliable_bicgstab(
    action: VectorAction,
    rhs: np.ndarray,
    x0: Optional[np.ndarray] = None,
    *,
    right_preconditioner: Optional[VectorAction] = None,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    residual_replacement_interval: Optional[int] = 25,
    max_breakdown_restarts: int = 2,
    breakdown_tolerance: Optional[float] = None,
) -> BiCGSTABResult:
    

    if not callable(action):
        raise TypeError("action must be callable")
    if right_preconditioner is not None and not callable(right_preconditioner):
        raise TypeError("right_preconditioner must be callable or None")

    rhs_array = np.asarray(rhs)
    if rhs_array.ndim != 1 or rhs_array.size == 0:
        raise ValueError("rhs must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(rhs_array)):
        raise FloatingPointError("rhs contains a non-finite value")

    if x0 is None:
        initial_dtype = rhs_array.dtype
    else:
        initial_array = np.asarray(x0)
        if initial_array.shape != rhs_array.shape:
            raise ValueError(
                f"x0 has shape {initial_array.shape}; expected {rhs_array.shape}"
            )
        if not np.all(np.isfinite(initial_array)):
            raise FloatingPointError("x0 contains a non-finite value")
        initial_dtype = np.result_type(rhs_array.dtype, initial_array.dtype)

    
    dtype = np.dtype(np.result_type(initial_dtype, np.complex128))
    b = np.array(rhs_array, dtype=dtype, copy=True)
    x = (
        np.zeros_like(b)
        if x0 is None
        else np.array(x0, dtype=dtype, copy=True)
    )

    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and nonnegative")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    if maxiter is None:
        maxiter = 2 * b.size
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    maxiter = int(maxiter)
    if residual_replacement_interval is not None:
        if (
            isinstance(residual_replacement_interval, bool)
            or int(residual_replacement_interval)
            != residual_replacement_interval
            or residual_replacement_interval < 0
        ):
            raise ValueError(
                "residual_replacement_interval must be a nonnegative integer "
                "or None"
            )
        residual_replacement_interval = int(residual_replacement_interval)
    if (
        isinstance(max_breakdown_restarts, bool)
        or int(max_breakdown_restarts) != max_breakdown_restarts
        or max_breakdown_restarts < 0
    ):
        raise ValueError("max_breakdown_restarts must be a nonnegative integer")
    max_breakdown_restarts = int(max_breakdown_restarts)
    if breakdown_tolerance is None:
        breakdown_tolerance = 100.0 * np.finfo(np.float64).eps
    if not np.isfinite(breakdown_tolerance) or breakdown_tolerance < 0.0:
        raise ValueError(
            "breakdown_tolerance must be finite and nonnegative"
        )
    breakdown_tolerance = float(breakdown_tolerance)

    matvecs = 0

    def apply(vector: np.ndarray) -> np.ndarray:
        nonlocal matvecs
        image = action(vector)
        matvecs += 1
        return _checked_vector(
            image, shape=b.shape, dtype=dtype, source="action"
        )

    def precondition(vector: np.ndarray) -> np.ndarray:
        if right_preconditioner is None:
            
            return vector.copy()
        return _checked_vector(
            right_preconditioner(vector),
            shape=b.shape,
            dtype=dtype,
            source="right_preconditioner",
        )

    rhs_norm = float(np.linalg.norm(b))
    residual_denominator = rhs_norm if rhs_norm > 0.0 else 1.0
    target = max(float(atol), float(rtol) * rhs_norm)

    
    r = b.copy() if x0 is None else b - apply(x)
    true_residual_norm = float(np.linalg.norm(r))
    residual_is_true = True

    def result(converged: bool, iterations: int, breakdowns: int):
        return BiCGSTABResult(
            solution=x.copy(),
            relative_residual=float(
                true_residual_norm / residual_denominator
            ),
            iterations=iterations,
            matvecs=matvecs,
            converged=converged,
            breakdowns=breakdowns,
        )

    if true_residual_norm <= target:
        return result(True, 0, 0)
    if maxiter == 0:
        return result(False, 0, 0)

    r_hat = _restart_shadow(r, 0)
    p = np.zeros_like(r)
    v = np.zeros_like(r)
    rho_previous = 1.0 + 0.0j
    alpha = 1.0 + 0.0j
    omega = 1.0 + 0.0j
    fresh_recurrence = True
    iterations = 0
    breakdowns = 0

    def certify_and_restart(*, after_breakdown: bool) -> tuple[bool, bool]:
        

        nonlocal r, r_hat, true_residual_norm, residual_is_true
        nonlocal fresh_recurrence, breakdowns

        r = b - apply(x)
        true_residual_norm = float(np.linalg.norm(r))
        residual_is_true = True
        if true_residual_norm <= target:
            return True, True

        if after_breakdown:
            breakdowns += 1
            if breakdowns > max_breakdown_restarts:
                return False, False
            r_hat = _restart_shadow(r, breakdowns)
        else:
            r_hat = _restart_shadow(r, 0)
        fresh_recurrence = True
        return False, True

    while iterations < maxiter:
        iterations += 1
        rho = complex(np.vdot(r_hat, r))
        rho_scale = float(np.linalg.norm(r_hat) * np.linalg.norm(r))
        if _near_breakdown(rho, rho_scale, breakdown_tolerance):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue

        if fresh_recurrence:
            p = r.copy()
            fresh_recurrence = False
        else:
            beta = (rho / rho_previous) * (alpha / omega)
            if not np.isfinite(beta.real) or not np.isfinite(beta.imag):
                converged, can_continue = certify_and_restart(
                    after_breakdown=True
                )
                if converged:
                    return result(True, iterations, breakdowns)
                if not can_continue:
                    return result(False, iterations, breakdowns)
                continue
            p = r + beta * (p - omega * v)

        preconditioned = precondition(p)
        v = apply(preconditioned)
        alpha_denominator = complex(np.vdot(r_hat, v))
        alpha_scale = float(np.linalg.norm(r_hat) * np.linalg.norm(v))
        if _near_breakdown(
            alpha_denominator, alpha_scale, breakdown_tolerance
        ):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue

        alpha = rho / alpha_denominator
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue

        
        x = x + alpha * preconditioned
        r = r - alpha * v
        residual_is_true = False
        recursive_norm = float(np.linalg.norm(r))
        if recursive_norm <= target:
            converged, can_continue = certify_and_restart(
                after_breakdown=False
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:  
                return result(False, iterations, breakdowns)
            continue

        preconditioned = precondition(r)
        t = apply(preconditioned)
        t_norm = float(np.linalg.norm(t))
        omega_numerator = complex(np.vdot(t, r))
        omega_scale = t_norm * recursive_norm
        if _near_breakdown(
            omega_numerator, omega_scale, breakdown_tolerance
        ):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue

        t_norm_squared = t_norm * t_norm
        if t_norm_squared == 0.0 or not np.isfinite(t_norm_squared):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue
        omega = omega_numerator / t_norm_squared
        if not np.isfinite(omega.real) or not np.isfinite(omega.imag):
            converged, can_continue = certify_and_restart(
                after_breakdown=True
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:
                return result(False, iterations, breakdowns)
            continue

        x = x + omega * preconditioned
        r = r - omega * t
        residual_is_true = False
        rho_previous = rho

        recursive_norm = float(np.linalg.norm(r))
        candidate_converged = recursive_norm <= target
        replacement_due = bool(
            residual_replacement_interval
            and iterations % residual_replacement_interval == 0
        )
        if candidate_converged or replacement_due:
            converged, can_continue = certify_and_restart(
                after_breakdown=False
            )
            if converged:
                return result(True, iterations, breakdowns)
            if not can_continue:  
                return result(False, iterations, breakdowns)

    
    if not residual_is_true:
        r = b - apply(x)
        true_residual_norm = float(np.linalg.norm(r))
        residual_is_true = True
    return result(true_residual_norm <= target, iterations, breakdowns)


__all__ = ["BiCGSTABResult", "reliable_bicgstab"]
