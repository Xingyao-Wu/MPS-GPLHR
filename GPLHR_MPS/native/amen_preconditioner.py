from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from renormalizer import Model, Mpo, Mps
from renormalizer.model import h_qc
from renormalizer.mps.matrix import asnumpy

from projects.feast_mps_hubbard_nonhermitian.amen_solver import (
    solve_shifted_amen,
)

LOCAL_QN = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int16)


def build_hubbard_mpo(
    parameters, *, similarity_balanced: bool = False
) -> tuple[Model, Mpo]:
    """Build the original or exactly similarity-balanced Hubbard MPO."""

    norbitals = 2 * parameters.sites
    one_body = np.zeros((norbitals, norbitals), dtype=np.float64)
    two_body = np.zeros(
        (norbitals, norbitals, norbitals, norbitals), dtype=np.float64
    )
    for site, potential in enumerate(parameters.onsite_potential):
        up, down = 2 * site, 2 * site + 1
        one_body[up, up] += potential
        one_body[down, down] += potential
        two_body[up, down, down, up] = parameters.interaction
        if site + 1 < parameters.sites:
            next_up, next_down = up + 2, down + 2
            if similarity_balanced:
                hopping_right = hopping_left = math.sqrt(
                    parameters.hopping_right * parameters.hopping_left
                )
            else:
                hopping_right = parameters.hopping_right
                hopping_left = parameters.hopping_left
            one_body[next_up, up] = -hopping_right
            one_body[next_down, down] = -hopping_right
            one_body[up, next_up] = -hopping_left
            one_body[down, next_down] = -hopping_left
    basis, terms = h_qc.qc_model(
        one_body,
        two_body,
        conserve_qn=True,
        spatial_orbital=True,
    )
    model = Model(basis, terms)
    return model, Mpo(model)


def renormalizer_sector_indices(basis: np.ndarray, sites: int) -> np.ndarray:
    """Map the fixed-sector ordering to Renormalizer's local basis ordering."""

    indices = np.zeros(len(basis), dtype=np.int64)
    for site in range(sites):
        up = ((basis >> np.uint32(2 * site)) & np.uint32(1)).astype(np.int64)
        down = (
            (basis >> np.uint32(2 * site + 1)) & np.uint32(1)
        ).astype(np.int64)
        indices = 4 * indices + up + 2 * down
    return indices


def _groups(labels: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    groups = {}
    for qn in np.unique(labels, axis=0):
        key = (int(qn[0]), int(qn[1]))
        groups[key] = np.flatnonzero(np.all(labels == qn, axis=1))
    return groups


def native_src_to_renormalizer(state, model: Model) -> Mps:
    """Convert native SRC blocks without reconstructing a full Fock vector."""

    if len(state.site_blocks) != model.nsite:
        raise ValueError("SRC state and AMEn model have different site counts")
    qn_to_physical = []
    for local_basis in model.basis:
        mapping = {
            tuple(int(value) for value in qn): physical
            for physical, qn in enumerate(local_basis.sigmaqn)
        }
        if set(mapping) != {tuple(int(value) for value in qn) for qn in LOCAL_QN}:
            raise ValueError("AMEn model uses an incompatible local QN basis")
        qn_to_physical.append(mapping)

    if hasattr(state, "bond_qns"):
        bond_qns = state.bond_qns
    else:
        bond_qns = []
        for dimensions in state.bond_dims:
            labels = []
            for qn, dimension in sorted(dimensions.items()):
                labels.extend([qn] * int(dimension))
            bond_qns.append(np.asarray(labels, dtype=np.int16))

    result = Mps()
    result.model = model
    result.dtype = np.complex128
    result.build_empty_mp(model.nsite)
    for site, blocks in enumerate(state.site_blocks):
        left_labels = bond_qns[site]
        right_labels = bond_qns[site + 1]
        left_groups = _groups(left_labels)
        right_groups = _groups(right_labels)
        core = np.zeros(
            (len(left_labels), 4, len(right_labels)), dtype=np.complex128
        )
        for (left_qn, cpu_physical, right_qn), block in blocks.items():
            physical_qn = tuple(int(value) for value in LOCAL_QN[cpu_physical])
            physical = qn_to_physical[site][physical_qn]
            core[np.ix_(
                left_groups[left_qn],
                [physical],
                right_groups[right_qn],
            )] = np.asarray(asnumpy(block))[:, None, :]
        result[site] = core

    
    result.qn = [labels.copy() for labels in bond_qns]
    result.qn[-1] = np.zeros_like(result.qn[-1])
    result.qntot = np.asarray(bond_qns[-1][0], dtype=int)
    result.qnidx = model.nsite - 1
    result.to_right = False
    result.coeff = 1.0 + 0.0j
    return result


def renormalizer_to_native(state: Mps):
    """Convert a left-canonical Reno MPS directly to native QN blocks."""

    from .native_tensor import NativeMPS

    result = state.copy()
    result.ensure_left_canonical()
    total_qn = tuple(int(value) for value in result.qntot)
    bond_qns = [np.asarray(labels, dtype=np.int16).copy() for labels in result.qn]
    bond_qns[-1] = np.asarray([total_qn], dtype=np.int16)
    bond_dims = []
    for labels in bond_qns:
        unique, counts = np.unique(labels, axis=0, return_counts=True)
        bond_dims.append(
            {
                (int(qn[0]), int(qn[1])): int(count)
                for qn, count in zip(unique, counts)
            }
        )

    site_blocks = []
    coefficient = complex(result.coeff)
    for site, tensor in enumerate(result):
        core = np.asarray(asnumpy(tensor.array), dtype=np.complex128)
        if site == 0:
            core = coefficient * core
        left_groups = _groups(bond_qns[site])
        right_groups = _groups(bond_qns[site + 1])
        physical_map = {
            tuple(int(value) for value in qn): physical
            for physical, qn in enumerate(result.model.basis[site].sigmaqn)
        }
        blocks = {}
        for native_physical, local_qn_array in enumerate(LOCAL_QN):
            local_qn = tuple(int(value) for value in local_qn_array)
            physical = physical_map[local_qn]
            for left_qn, left_indices in left_groups.items():
                right_qn = (
                    left_qn[0] + local_qn[0],
                    left_qn[1] + local_qn[1],
                )
                right_indices = right_groups.get(right_qn)
                if right_indices is None:
                    continue
                block = core[np.ix_(left_indices, [physical], right_indices)][:, 0, :]
                if np.any(block != 0.0):
                    blocks[(left_qn, native_physical, right_qn)] = block
        site_blocks.append(blocks)
    return NativeMPS(site_blocks, bond_dims, total_qn)


def mps_sector_vector(state: Mps, sector_indices: np.ndarray) -> np.ndarray:
    """Contract selected fixed-sector configurations without a full vector."""

    indices = np.asarray(sector_indices, dtype=np.int64)
    digits = [
        (indices // (4 ** (state.site_num - site - 1))) % 4
        for site in range(state.site_num)
    ]
    amplitudes = np.ones((indices.size, 1), dtype=np.complex128)
    for site, tensor in enumerate(state):
        core = np.asarray(tensor.array)
        updated = np.zeros((indices.size, core.shape[-1]), dtype=np.complex128)
        for physical in range(core.shape[1]):
            selected = np.flatnonzero(digits[site] == physical)
            if selected.size:
                updated[selected] = amplitudes[selected] @ core[:, physical, :]
        amplitudes = updated
    return complex(state.coeff) * amplitudes[:, 0]


def maximum_qn_degeneracy(state: Mps) -> int:
    maximum = 1
    for labels in state.qn:
        _unique, counts = np.unique(np.asarray(labels), axis=0, return_counts=True)
        maximum = max(maximum, int(np.max(counts)))
    return maximum


def apply_hubbard_similarity(
    state: Mps, parameters, *, inverse: bool
) -> Mps:
    """Apply the exact bond-one map between physical and balanced Hubbard forms.

    ``inverse=True`` maps a physical vector to the balanced Hermitian frame.
    It includes both asymmetric-hopping and onsite-Jastrow similarities.
    """

    result = state.copy()
    ratio = math.sqrt(parameters.hopping_right / parameters.hopping_left)
    center = 0.5 * (parameters.sites - 1)
    sign = -1.0 if inverse else 1.0
    jastrow_sign = 1.0 if inverse else -1.0
    for site, local_basis in enumerate(result.model.basis):
        local_qns = np.asarray(local_basis.sigmaqn)
        occupations = np.sum(local_qns, axis=1)
        double_occupancy = np.prod(local_qns, axis=1)
        weights = ratio ** (sign * (site - center) * occupations)
        weights *= np.exp(
            jastrow_sign
            * float(parameters.onsite_jastrow_strength)
            * double_occupancy
        )
        result[site] = np.asarray(result[site].array) * weights[None, :, None]
    return result


@dataclass(frozen=True)
class CheapAMEnSettings:
    max_sweeps: int = 2
    local_maxiter: int = 4
    local_tolerance: float = 5.0e-2
    enrichment_rank: int = 1
    residual_replacement_interval: int = 4
    half_sweeps_per_sweep: int = 2
    use_block_env: bool = True


class CheapAMEnPreconditioner:
    """Apply a deliberately inexact shifted inverse with warm starts."""

    def __init__(
        self,
        parameters,
        basis: np.ndarray,
        sector_matrix,
        *,
        target: complex,
        project_max_bond: int,
        settings: CheapAMEnSettings,
        seed: int,
    ):
        self.parameters = parameters
        self.model, self.hamiltonian = build_hubbard_mpo(
            parameters, similarity_balanced=True
        )
        self.sector_indices = renormalizer_sector_indices(
            basis, parameters.sites
        )
        self.sector_matrix = sector_matrix
        self.target = complex(target)
        self.project_max_bond = int(project_max_bond)
        self.settings = settings
        self.seed = int(seed)
        self.warm_starts: dict[str, list[Mps]] = {}
        self.calls = 0

    def _initial_guess(self, root: int) -> Mps:
        np.random.seed(self.seed + 1009 * self.calls + 37 * root)
        guess = Mps.random(
            self.model,
            [self.parameters.n_up, self.parameters.n_down],
            self.project_max_bond,
            percent=1.0,
        )
        guess.to_complex(inplace=True)
        return guess

    def apply(
        self,
        source: np.ndarray,
        rhs_states: list,
        *,
        stage: str,
        max_sweeps: int | None = None,
        initial_states: list | None = None,
    ) -> tuple[np.ndarray, dict]:
        source = np.asarray(source, dtype=np.complex128)
        if len(rhs_states) != source.shape[1]:
            raise ValueError("AMEn RHS state count does not match source block")
        if initial_states is not None and len(initial_states) != source.shape[1]:
            raise ValueError("AMEn initial-state count does not match source block")
        old_warm = self.warm_starts.get(stage, [])
        new_warm = []
        outputs = []
        reports = []
        for root, rhs_native in enumerate(rhs_states):
            rhs = native_src_to_renormalizer(rhs_native, self.model)
            rhs = apply_hubbard_similarity(
                rhs, self.parameters, inverse=True
            )
            # solve (sigma I-H)x=-f, which is (H-sigma I)x=f
            rhs.scale(-1.0, inplace=True)
            if root < len(old_warm):
                x0 = old_warm[root].copy()
                initial_mode = "same-stage AMEn warm start"
            elif initial_states is not None:
                x0 = native_src_to_renormalizer(
                    initial_states[root], self.model
                )
                x0 = apply_hubbard_similarity(
                    x0, self.parameters, inverse=True
                )
                x0.scale(1.0 / max(abs(self.target), 1.0), inplace=True)
                initial_mode = "SRC approximation of current RHS"
            else:
                x0 = self._initial_guess(root)
                initial_mode = "small random QN MPS"
            if maximum_qn_degeneracy(x0) > self.project_max_bond:
                raise RuntimeError(
                    "SRC AMEn initial guess exceeds the symmetry-block bond cap"
                )
            solution, report = solve_shifted_amen(
                self.hamiltonian,
                rhs,
                self.target,
                x0=x0,
                max_sweeps=(
                    self.settings.max_sweeps
                    if max_sweeps is None
                    else int(max_sweeps)
                ),
                local_maxiter=self.settings.local_maxiter,
                local_tolerance=self.settings.local_tolerance,
                bond_cap=self.project_max_bond,
                cutoff=1.0e-12,
                global_tolerance=0.5,
                enrichment_rank=self.settings.enrichment_rank,
                residual_replacement_interval=(
                    self.settings.residual_replacement_interval
                ),
                stagnation_sweeps=1,
                use_block_env=self.settings.use_block_env,
                half_sweeps_per_sweep=(
                    self.settings.half_sweeps_per_sweep
                ),
                certify_global_residual=False,
                symmetry_block_bond_cap=self.project_max_bond,
            )
            physical_solution = apply_hubbard_similarity(
                solution, self.parameters, inverse=False
            )
            physical_solution.ensure_left_canonical()
            sector_solution = mps_sector_vector(
                physical_solution, self.sector_indices
            )
            relative = float(
                np.linalg.norm(
                    self.sector_matrix @ sector_solution
                    - self.target * sector_solution
                    - source[:, root]
                )
                / max(np.linalg.norm(source[:, root]), 1.0e-300)
            )
            sector_bond = maximum_qn_degeneracy(solution)
            if sector_bond > self.project_max_bond:
                raise RuntimeError("AMEn exceeded the project symmetry-block cap")
            if not np.array_equal(
                solution.qntot,
                [self.parameters.n_up, self.parameters.n_down],
            ):
                raise RuntimeError("AMEn left the requested U(1)xU(1) sector")
            outputs.append(sector_solution)
            new_warm.append(solution.copy())
            row = asdict(report)
            row["shift"] = {
                "real": float(report.shift.real),
                "imag": float(report.shift.imag),
            }
            row.update(
                verified_original_relative_residual=relative,
                initial_mode=initial_mode,
                similarity_balanced=True,
                similarity_map_bond_dimension=1,
                maximum_symmetry_block_bond=sector_bond,
                total_bond_dimensions=[
                    int(value) for value in solution.bond_dims
                ],
            )
            reports.append(row)
        self.warm_starts[stage] = new_warm
        self.calls += 1
        return np.stack(outputs, axis=1), {
            "type": "cheap inexact one-site AMEn",
            "stage": stage,
            "settings": asdict(self.settings),
            "columns": reports,
            "maximum_verified_original_relative_residual": max(
                row["verified_original_relative_residual"] for row in reports
            ),
            "total_half_sweeps": sum(row["half_sweeps"] for row in reports),
            "total_local_matvecs": sum(row["local_matvecs"] for row in reports),
            "seconds": sum(row["seconds"] for row in reports),
        }


__all__ = [
    "CheapAMEnPreconditioner",
    "CheapAMEnSettings",
    "apply_hubbard_similarity",
    "build_hubbard_mpo",
    "maximum_qn_degeneracy",
    "mps_sector_vector",
    "native_src_to_renormalizer",
    "renormalizer_to_native",
    "renormalizer_sector_indices",
]
