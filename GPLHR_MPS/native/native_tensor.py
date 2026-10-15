from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os
import time
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg

from renormalizer.mps.backend import GPU_ID, USE_GPU, xp
from renormalizer.mps.matrix import asnumpy, asxp


Qn = tuple[int, int]
BlockKey = tuple[Qn, int, Qn]
LOCAL_QN: tuple[Qn, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


_BACKEND_COUNTERS = {
    "gpu_pivoted_qr_calls": 0,
    "gpu_exact_unpivoted_qr_calls": 0,
    "cpu_pivoted_qr_fallback_calls": 0,
    "batched_contraction_groups": 0,
    "batched_contraction_terms": 0,
    "grouped_qr_launches": 0,
    "grouped_qr_matrices": 0,
    "grouped_svd_launches": 0,
    "grouped_svd_matrices": 0,
    "grouped_cholesky_qr_launches": 0,
    "grouped_cholesky_qr_matrices": 0,
    "grouped_cholesky_qr_fallbacks": 0,
    "padded_qr_launches": 0,
    "padded_qr_matrices": 0,
    "batched_src_groups": 0,
    "batched_src_blocks": 0,
    "padded_src_launches": 0,
    "padded_src_blocks": 0,
}
_GPU_PIVOTED_QR_AVAILABLE: bool | None = None
_EAGER_FINITE_CHECK = (
    not USE_GPU
    or os.environ.get("GPLHR_GPU_EAGER_FINITE_CHECK", "0").lower()
    in {"1", "true", "yes", "on"}
)


def _scalar_float(value) -> float:
    return float(np.asarray(asnumpy(value) if USE_GPU else value))


def _scalar_complex(value) -> complex:
    return complex(np.asarray(asnumpy(value) if USE_GPU else value))


def _scalar_bool(value) -> bool:
    return bool(np.asarray(asnumpy(value) if USE_GPU else value))


def _arrays_are_numerically_real(arrays: Iterable[object]) -> bool:
    """Classify a collection with one device synchronization, not per block."""

    arrays = list(arrays)
    if not arrays:
        return True
    total = xp.sum(
        xp.stack([xp.sum(xp.abs(array) ** 2) for array in arrays])
    )
    imaginary = xp.sum(
        xp.stack([xp.sum(xp.imag(array) ** 2) for array in arrays])
    )
    totals = np.asarray(asnumpy(xp.stack([total, imaginary])), dtype=np.float64)
    scale = max(math.sqrt(max(totals[0], 0.0)), 1.0)
    return math.sqrt(max(totals[1], 0.0)) <= 1.0e-12 * scale


def _grouped_reduced_qr(matrices: Sequence[object]):
    """Run reduced QR in shape-homogeneous batches and restore input order."""

    outputs = [None] * len(matrices)
    groups = defaultdict(list)
    for index, matrix in enumerate(matrices):
        groups[(matrix.shape, matrix.dtype.str)].append(index)
    for indices in groups.values():
        batch = xp.stack([matrices[index] for index in indices])
        basis, triangular = xp.linalg.qr(batch, mode="reduced")
        _BACKEND_COUNTERS["grouped_qr_launches"] += 1
        _BACKEND_COUNTERS["grouped_qr_matrices"] += len(indices)
        for batch_index, output_index in enumerate(indices):
            outputs[output_index] = (basis[batch_index], triangular[batch_index])
    return outputs


def _grouped_right_factorization(matrices: Sequence[object]):
    """Factor ``A = transfer @ right`` using grouped exact GPU kernels."""

    method = os.environ.get("GPLHR_EXACT_GAUGE", "qr").lower()
    if method not in {"qr", "padded_qr", "svd", "cholqr"}:
        raise ValueError(
            "GPLHR_EXACT_GAUGE must be 'qr', 'padded_qr', 'svd', or 'cholqr'"
        )
    if method == "padded_qr":
        transposed = [matrix.conj().T for matrix in matrices]
        maximum_rows = max(matrix.shape[0] for matrix in transposed)
        maximum_columns = max(matrix.shape[1] for matrix in transposed)
        batch = xp.zeros(
            (len(transposed), maximum_rows, maximum_columns),
            dtype=transposed[0].dtype,
        )
        for index, matrix in enumerate(transposed):
            batch[index, : matrix.shape[0], : matrix.shape[1]] = matrix
        basis, triangular = xp.linalg.qr(batch, mode="reduced")
        _BACKEND_COUNTERS["padded_qr_launches"] += 1
        _BACKEND_COUNTERS["padded_qr_matrices"] += len(matrices)
        outputs = []
        for index, matrix in enumerate(transposed):
            rank = min(matrix.shape)
            outputs.append(
                (
                    basis[index, : matrix.shape[0], :rank].conj().T,
                    triangular[index, :rank, : matrix.shape[1]].conj().T,
                )
            )
        return outputs
    if method == "qr":
        qr_outputs = _grouped_reduced_qr(
            [matrix.conj().T for matrix in matrices]
        )
        return [
            (basis.conj().T, triangular.conj().T)
            for basis, triangular in qr_outputs
        ]

    outputs = [None] * len(matrices)
    groups = defaultdict(list)
    for index, matrix in enumerate(matrices):
        groups[(matrix.shape, matrix.dtype.str)].append(index)
    for indices in groups.values():
        batch = xp.stack([matrices[index] for index in indices])
        if method == "cholqr" and batch.shape[1] <= batch.shape[2]:
            try:
                gram = batch @ batch.conj().transpose(0, 2, 1)
                transfer = xp.linalg.cholesky(gram)
                right = xp.linalg.solve(transfer, batch)
                _BACKEND_COUNTERS["grouped_cholesky_qr_launches"] += 1
                _BACKEND_COUNTERS["grouped_cholesky_qr_matrices"] += len(indices)
                for batch_index, output_index in enumerate(indices):
                    outputs[output_index] = (
                        right[batch_index],
                        transfer[batch_index],
                    )
                continue
            except np.linalg.LinAlgError:
                _BACKEND_COUNTERS["grouped_cholesky_qr_fallbacks"] += len(indices)
                qr_outputs = _grouped_reduced_qr(
                    [matrices[index].conj().T for index in indices]
                )
                for (basis, triangular), output_index in zip(
                    qr_outputs, indices
                ):
                    outputs[output_index] = (
                        basis.conj().T,
                        triangular.conj().T,
                    )
                continue
        if method == "cholqr":
            qr_outputs = _grouped_reduced_qr(
                [matrices[index].conj().T for index in indices]
            )
            for (basis, triangular), output_index in zip(qr_outputs, indices):
                outputs[output_index] = (
                    basis.conj().T,
                    triangular.conj().T,
                )
            continue
        left, singular, right = xp.linalg.svd(batch, full_matrices=False)
        transfer = left * singular[:, None, :]
        _BACKEND_COUNTERS["grouped_svd_launches"] += 1
        _BACKEND_COUNTERS["grouped_svd_matrices"] += len(indices)
        for batch_index, output_index in enumerate(indices):
            outputs[output_index] = (
                right[batch_index],
                transfer[batch_index],
            )
    return outputs


def gpu_parallel_map(function, arguments: Sequence[object]):
    """Run independent tensor tasks on separate CUDA streams.

    Each task still operates on symmetry-labelled MPS blocks.  Parallelism is
    only across independent GPLHR columns; it never merges physical sectors
    or changes the SRC truncation rule.
    """

    if not USE_GPU or len(arguments) < 2:
        return [function(argument) for argument in arguments]
    workers = min(
        len(arguments),
        max(1, int(os.environ.get("GPLHR_GPU_STREAMS", "1"))),
    )
    if workers == 1:
        return [function(argument) for argument in arguments]

    def run(argument):
        with xp.cuda.Device(GPU_ID):
            stream = xp.cuda.Stream(non_blocking=True)
            with stream:
                result = function(argument)
            stream.synchronize()
            return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, arguments))


def _pivoted_qr(matrix):
    """Rank-revealing QR on GPU when CuPy implements pivoting.

    CuPy versions without pivoted QR fall back for this exact gauge operation
    only.  The returned factors are moved back to the active tensor backend;
    all subsequent block contractions and lossy SRC operations stay there.
    """

    global _GPU_PIVOTED_QR_AVAILABLE
    if USE_GPU and _GPU_PIVOTED_QR_AVAILABLE is not False:
        try:
            import cupyx.scipy.linalg as cupy_linalg

            result = cupy_linalg.qr(
                matrix,
                mode="economic",
                pivoting=True,
                check_finite=False,
            )
            _GPU_PIVOTED_QR_AVAILABLE = True
            _BACKEND_COUNTERS["gpu_pivoted_qr_calls"] += 1
            return result
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            _GPU_PIVOTED_QR_AVAILABLE = False

    basis, triangular, pivots = scipy.linalg.qr(
        asnumpy(matrix),
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    if USE_GPU:
        _BACKEND_COUNTERS["cpu_pivoted_qr_fallback_calls"] += 1
    return asxp(basis), asxp(triangular), asxp(pivots)


def native_backend_info() -> dict:
    """Describe where native tensor work is executed."""

    info = {
        "array_backend": "cupy" if USE_GPU else "numpy",
        "device": "cuda" if USE_GPU else "cpu",
        "gpu_id": None if not USE_GPU else int(GPU_ID),
        "small_projected_algebra": "numpy/scipy on CPU",
        "symmetry_metadata": "CPU charge maps",
        "exact_right_canonicalization": (
            "backend-native exact grouped "
            + os.environ.get("GPLHR_EXACT_GAUGE", "qr").upper()
            + " without null pruning"
        ),
        "eager_full_tensor_finite_check": bool(_EAGER_FINITE_CHECK),
        "pivoted_qr_api": (
            "cupyx.scipy.linalg"
            if _GPU_PIVOTED_QR_AVAILABLE is True
            else "scipy CPU fallback"
            if USE_GPU and _GPU_PIVOTED_QR_AVAILABLE is False
            else "scipy"
        ),
        **_BACKEND_COUNTERS,
    }
    if USE_GPU:
        pool = xp.get_default_memory_pool()
        info.update(
            gpu_memory_used_bytes=int(pool.used_bytes()),
            gpu_memory_reserved_bytes=int(pool.total_bytes()),
        )
    return info


def add_qn(left: Qn, right: Qn) -> Qn:
    return left[0] + right[0], left[1] + right[1]


def _qn(row: Iterable[int]) -> Qn:
    values = tuple(int(value) for value in row)
    if len(values) != 2:
        raise ValueError("U(1)xU(1) requires two charge components")
    return values


@dataclass
class NativeMPS:
    """MPS storing only charge-allowed numerical blocks."""

    site_blocks: list[dict[BlockKey, object]]
    bond_dims: list[dict[Qn, int]]
    total_qn: Qn
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.site_blocks = [
            {
                key: xp.ascontiguousarray(asxp(value), dtype=xp.complex128)
                for key, value in blocks.items()
                if value.size
            }
            for blocks in self.site_blocks
        ]
        self.bond_dims = [
            {tuple(key): int(value) for key, value in dims.items() if value > 0}
            for dims in self.bond_dims
        ]
        self.validate()

    @property
    def nsites(self) -> int:
        return len(self.site_blocks)

    @property
    def max_bond(self) -> int:
        """Largest conventional, sector-summed virtual-bond dimension."""

        return max(self.bond_dimensions)

    @property
    def max_charge_block_bond(self) -> int:
        """Largest retained degeneracy inside one charge sector."""

        return max(max(dims.values(), default=0) for dims in self.bond_dims)

    @property
    def max_total_bond(self) -> int:
        """Explicit alias for the conventional sector-summed dimension."""

        return self.max_bond

    @property
    def bond_dimensions(self) -> list[int]:
        """Conventional dimensions of the virtual bonds."""

        return [sum(dims.values()) for dims in self.bond_dims]

    @property
    def stored_elements(self) -> int:
        return sum(block.size for site in self.site_blocks for block in site.values())

    def copy(self) -> "NativeMPS":
        return NativeMPS(
            [
                {key: value.copy() for key, value in blocks.items()}
                for blocks in self.site_blocks
            ],
            [dict(dims) for dims in self.bond_dims],
            self.total_qn,
            dict(self.metadata),
        )

    def validate(self, max_bond: int | None = None) -> None:
        if not self.site_blocks or len(self.bond_dims) != self.nsites + 1:
            raise ValueError("invalid native MPS layout")
        if self.bond_dims[0] != {(0, 0): 1}:
            raise ValueError("left MPS boundary must have charge (0,0)")
        if self.bond_dims[-1] != {self.total_qn: 1}:
            raise ValueError("right MPS boundary has the wrong total charge")
        if max_bond is not None and self.max_bond > max_bond:
            raise RuntimeError("native MPS exceeded the sector-summed max_bond cap")
        finite_flags = []
        for site, blocks in enumerate(self.site_blocks):
            for (left_qn, physical, right_qn), block in blocks.items():
                if not 0 <= physical < len(LOCAL_QN):
                    raise ValueError("invalid local physical index")
                if add_qn(left_qn, LOCAL_QN[physical]) != right_qn:
                    raise ValueError("symmetry-forbidden MPS block")
                expected = (
                    self.bond_dims[site].get(left_qn, 0),
                    self.bond_dims[site + 1].get(right_qn, 0),
                )
                if block.shape != expected:
                    raise ValueError(
                        f"site {site} block {left_qn, physical, right_qn} "
                        f"has {block.shape}, expected {expected}"
                    )
                if _EAGER_FINITE_CHECK:
                    finite_flags.append(xp.all(xp.isfinite(block)))
        if finite_flags and not _scalar_bool(xp.all(xp.stack(finite_flags))):
            raise FloatingPointError("nonfinite native MPS block")

    def scale(self, factor: complex) -> "NativeMPS":
        result = self.copy()
        for key in result.site_blocks[0]:
            result.site_blocks[0][key] *= factor
        return result


@dataclass(frozen=True)
class MPOEntry:
    left: int
    physical_out: int
    physical_in: int
    right: int
    value: complex


@dataclass
class NativeMPO:
    """A charge-labelled MPO stored as nonzero scalar transitions."""

    entries: list[list[MPOEntry]]
    bond_qns: list[list[Qn]]

    @property
    def nsites(self) -> int:
        return len(self.entries)

    @property
    def bond_dimensions(self) -> list[int]:
        return [len(labels) for labels in self.bond_qns]

    @classmethod
    def from_renormalizer(cls, model, mpo, tolerance: float = 1.0e-14) -> "NativeMPO":
        """Copy a small Renormalizer MPO into sparse transition storage."""

        desired = {qn: index for index, qn in enumerate(LOCAL_QN)}
        site_entries: list[list[MPOEntry]] = []
        for site, tensor in enumerate(mpo):
            array = np.asarray(asnumpy(tensor.array), dtype=np.complex128)
            physical_map = [desired[_qn(row)] for row in model.basis[site].sigmaqn]
            entries = []
            nonzero = np.argwhere(np.abs(array) > tolerance)
            for left, pout_model, pin_model, right in nonzero:
                entries.append(
                    MPOEntry(
                        int(left),
                        physical_map[int(pout_model)],
                        physical_map[int(pin_model)],
                        int(right),
                        complex(array[left, pout_model, pin_model, right]),
                    )
                )
            site_entries.append(entries)
        bond_qns = [[_qn(row) for row in labels] for labels in mpo.qn]
        result = cls(site_entries, bond_qns)
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.bond_qns) != self.nsites + 1:
            raise ValueError("invalid native MPO layout")
        if self.bond_dimensions[0] != 1 or self.bond_dimensions[-1] != 1:
            raise ValueError("native MPO must have scalar boundaries")
        for site, entries in enumerate(self.entries):
            for entry in entries:
                left_qn = self.bond_qns[site][entry.left]
                right_qn = self.bond_qns[site + 1][entry.right]
                delta = (
                    LOCAL_QN[entry.physical_out][0]
                    - LOCAL_QN[entry.physical_in][0],
                    LOCAL_QN[entry.physical_out][1]
                    - LOCAL_QN[entry.physical_in][1],
                )
                if add_qn(left_qn, delta) != right_qn:
                    raise ValueError("symmetry-forbidden MPO transition")


def diagonal_similarity_mpo(
    operator: NativeMPO,
    local_diagonals: Sequence[Sequence[complex]],
) -> NativeMPO:
    """Return ``S^-1 H S`` for a bond-one diagonal product ``S``.

    Transforming the physical input/output legs of every MPO core is exact.
    In particular, it retains all density-assisted and three-body correlated
    hopping induced by a local transcorrelator without expanding them into a
    hand-written fermionic operator list.
    """

    if len(local_diagonals) != operator.nsites:
        raise ValueError("diagonal similarity and MPO have different site counts")
    diagonals = []
    for diagonal in local_diagonals:
        values = np.asarray(diagonal, dtype=np.complex128)
        if values.shape != (len(LOCAL_QN),):
            raise ValueError("each local similarity diagonal needs four entries")
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) == 0.0):
            raise ValueError("local similarity diagonals must be finite and invertible")
        diagonals.append(values)
    entries = []
    for site, site_entries in enumerate(operator.entries):
        diagonal = diagonals[site]
        entries.append(
            [
                MPOEntry(
                    entry.left,
                    entry.physical_out,
                    entry.physical_in,
                    entry.right,
                    entry.value
                    * diagonal[entry.physical_in]
                    / diagonal[entry.physical_out],
                )
                for entry in site_entries
            ]
        )
    result = NativeMPO(entries, [list(labels) for labels in operator.bond_qns])
    result.validate()
    return result


def overlap(left: NativeMPS, right: NativeMPS) -> complex:
    if left.total_qn != right.total_qn or left.nsites != right.nsites:
        return 0.0j
    environments: dict[Qn, object] = {
        (0, 0): xp.ones((1, 1), dtype=xp.complex128)
    }
    for site in range(left.nsites):
        pending: dict[Qn, object] = {}
        left_by = defaultdict(list)
        right_by = defaultdict(list)
        for (ql, physical, qr), block in left.site_blocks[site].items():
            left_by[(ql, physical)].append((qr, block))
        for (ql, physical, qr), block in right.site_blocks[site].items():
            right_by[(ql, physical)].append((qr, block))
        for ql, environment in environments.items():
            for physical in range(4):
                for qr_left, block_left in left_by.get((ql, physical), ()):
                    for qr_right, block_right in right_by.get((ql, physical), ()):
                        if qr_left != qr_right:
                            continue
                        contribution = block_left.conj().T @ environment @ block_right
                        pending[qr_left] = pending.get(qr_left, 0.0) + contribution
        environments = pending
    final = environments.get(left.total_qn)
    return 0.0j if final is None else _scalar_complex(final[0, 0])


def norm(state: NativeMPS) -> float:
    return math.sqrt(max(float(overlap(state, state).real), 0.0))


def normalize(state: NativeMPS) -> NativeMPS:
    value = norm(state)
    if value <= 1.0e-300:
        raise FloatingPointError("cannot normalize a zero MPS")
    return state.scale(1.0 / value)


def action_inner(bra: NativeMPS, operator: NativeMPO, ket: NativeMPS) -> complex:
    """Contract ``<bra|operator|ket>`` without an action MPS."""

    if bra.nsites != operator.nsites or ket.nsites != operator.nsites:
        raise ValueError("MPO/MPS site mismatch")
    environments: dict[tuple[Qn, int, Qn], object] = {
        ((0, 0), 0, (0, 0)): xp.ones((1, 1), dtype=xp.complex128)
    }
    for site in range(operator.nsites):
        pending: dict[tuple[Qn, int, Qn], object] = {}
        bra_by = defaultdict(list)
        ket_by = defaultdict(list)
        op_by = defaultdict(list)
        for (ql, physical, qr), block in bra.site_blocks[site].items():
            bra_by[(ql, physical)].append((qr, block))
        for (ql, physical, qr), block in ket.site_blocks[site].items():
            ket_by[(ql, physical)].append((qr, block))
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        for (qbra, left_mpo, qket), environment in environments.items():
            for entry in op_by.get(left_mpo, ()):
                for qbra_r, bra_block in bra_by.get(
                    (qbra, entry.physical_out), ()
                ):
                    for qket_r, ket_block in ket_by.get(
                        (qket, entry.physical_in), ()
                    ):
                        key = (qbra_r, entry.right, qket_r)
                        contribution = (
                            entry.value
                            * (bra_block.conj().T @ environment @ ket_block)
                        )
                        pending[key] = pending.get(key, 0.0) + contribution
        environments = pending
    key = (bra.total_qn, 0, ket.total_qn)
    final = environments.get(key)
    return 0.0j if final is None else _scalar_complex(final[0, 0])


def double_action_inner(
    bra: NativeMPS, operator: NativeMPO, ket: NativeMPS
) -> complex:
    """Contract ``<H bra|H ket>`` as one double MPO layer."""

    environments: dict[tuple[Qn, int, int, Qn], object] = {
        ((0, 0), 0, 0, (0, 0)): xp.ones((1, 1), dtype=xp.complex128)
    }
    for site in range(operator.nsites):
        pending: dict[tuple[Qn, int, int, Qn], object] = {}
        bra_by = defaultdict(list)
        ket_by = defaultdict(list)
        op_by = defaultdict(list)
        for (ql, physical, qr), block in bra.site_blocks[site].items():
            bra_by[(ql, physical)].append((qr, block))
        for (ql, physical, qr), block in ket.site_blocks[site].items():
            ket_by[(ql, physical)].append((qr, block))
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        for (qbra, left_a, left_b, qket), environment in environments.items():
            entries_a = op_by.get(left_a, ())
            entries_b = op_by.get(left_b, ())
            for entry_a in entries_a:
                for entry_b in entries_b:
                    if entry_a.physical_out != entry_b.physical_out:
                        continue
                    for qbra_r, bra_block in bra_by.get(
                        (qbra, entry_a.physical_in), ()
                    ):
                        for qket_r, ket_block in ket_by.get(
                            (qket, entry_b.physical_in), ()
                        ):
                            key = (qbra_r, entry_a.right, entry_b.right, qket_r)
                            factor = entry_a.value.conjugate() * entry_b.value
                            contribution = factor * (
                                bra_block.conj().T @ environment @ ket_block
                            )
                            pending[key] = pending.get(key, 0.0) + contribution
        environments = pending
    key = (bra.total_qn, 0, 0, ket.total_qn)
    final = environments.get(key)
    return 0.0j if final is None else _scalar_complex(final[0, 0])


def _batched_accumulate(contributions):
    """Evaluate many allowed block triples with grouped batched GEMMs."""

    groups = defaultdict(list)
    for destination, factor, left, environment, right in contributions:
        signature = (
            left.shape,
            environment.shape,
            right.shape,
            left.dtype.str,
            environment.dtype.str,
            right.dtype.str,
        )
        groups[signature].append(
            (destination, factor, left, environment, right)
        )

    pending = {}
    for items in groups.values():
        left = xp.stack([item[2] for item in items], axis=0)
        environment = xp.stack([item[3] for item in items], axis=0)
        right = xp.stack([item[4] for item in items], axis=0)
        factors = xp.asarray([item[1] for item in items], dtype=xp.complex128)
        contracted = xp.matmul(
            xp.matmul(left.conj().transpose(0, 2, 1), environment),
            right,
        )
        contracted *= factors[:, None, None]
        indices_by_destination = defaultdict(list)
        for index, item in enumerate(items):
            indices_by_destination[item[0]].append(index)
        for destination, indices in indices_by_destination.items():
            selected = contracted[asxp(np.asarray(indices, dtype=np.int64))]
            value = xp.sum(selected, axis=0)
            pending[destination] = pending.get(destination, 0.0) + value
        _BACKEND_COUNTERS["batched_contraction_groups"] += 1
        _BACKEND_COUNTERS["batched_contraction_terms"] += len(items)
    return pending


def _collective_overlap_matrix(
    left_states: Sequence[NativeMPS],
    right_states: Sequence[NativeMPS],
) -> np.ndarray:
    """Contract every rectangular overlap using batched allowed blocks."""

    if not left_states or not right_states:
        return np.empty((len(left_states), len(right_states)), dtype=np.complex128)
    nsites = left_states[0].nsites
    total_qn = left_states[0].total_qn
    if any(state.nsites != nsites or state.total_qn != total_qn for state in (*left_states, *right_states)):
        raise ValueError("incompatible collective overlap block")
    environments = {
        (row, column, (0, 0)): xp.ones((1, 1), dtype=xp.complex128)
        for row in range(len(left_states))
        for column in range(len(right_states))
    }
    for site in range(nsites):
        left_by = defaultdict(list)
        right_by = defaultdict(list)
        for row, state in enumerate(left_states):
            for (ql, physical, qr), block in state.site_blocks[site].items():
                left_by[(row, ql, physical)].append((qr, block))
        for column, state in enumerate(right_states):
            for (ql, physical, qr), block in state.site_blocks[site].items():
                right_by[(column, ql, physical)].append((qr, block))
        contributions = []
        for (row, column, ql), environment in environments.items():
            for physical in range(4):
                for qr_left, block_left in left_by.get((row, ql, physical), ()):
                    for qr_right, block_right in right_by.get((column, ql, physical), ()):
                        if qr_left == qr_right:
                            contributions.append(
                                (
                                    (row, column, qr_left),
                                    1.0,
                                    block_left,
                                    environment,
                                    block_right,
                                )
                            )
        environments = _batched_accumulate(contributions)
    result = np.empty((len(left_states), len(right_states)), dtype=np.complex128)
    for row in range(len(left_states)):
        for column in range(len(right_states)):
            final = environments.get((row, column, total_qn))
            result[row, column] = 0.0j if final is None else _scalar_complex(final[0, 0])
    return result


def _collective_action_matrix(
    states: Sequence[NativeMPS], operator: NativeMPO
) -> np.ndarray:
    """Contract all ``<state_i|H|state_j>`` entries collectively."""

    size = len(states)
    environments = {
        (row, column, (0, 0), 0, (0, 0)): xp.ones(
            (1, 1), dtype=xp.complex128
        )
        for row in range(size)
        for column in range(size)
    }
    for site in range(operator.nsites):
        state_by = defaultdict(list)
        for index, state in enumerate(states):
            for (ql, physical, qr), block in state.site_blocks[site].items():
                state_by[(index, ql, physical)].append((qr, block))
        op_by = defaultdict(list)
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        contributions = []
        for (row, column, qbra, left_mpo, qket), environment in environments.items():
            for entry in op_by.get(left_mpo, ()):
                for qbra_r, bra_block in state_by.get(
                    (row, qbra, entry.physical_out), ()
                ):
                    for qket_r, ket_block in state_by.get(
                        (column, qket, entry.physical_in), ()
                    ):
                        contributions.append(
                            (
                                (row, column, qbra_r, entry.right, qket_r),
                                entry.value,
                                bra_block,
                                environment,
                                ket_block,
                            )
                        )
        environments = _batched_accumulate(contributions)
    result = np.empty((size, size), dtype=np.complex128)
    total_qn = states[0].total_qn
    for row in range(size):
        for column in range(size):
            final = environments.get((row, column, total_qn, 0, total_qn))
            result[row, column] = 0.0j if final is None else _scalar_complex(final[0, 0])
    return result


def _collective_double_action_matrix(
    states: Sequence[NativeMPS], operator: NativeMPO
) -> np.ndarray:
    """Contract all ``<H state_i|H state_j>`` entries collectively."""

    size = len(states)
    environments = {
        (row, column, (0, 0), 0, 0, (0, 0)): xp.ones(
            (1, 1), dtype=xp.complex128
        )
        for row in range(size)
        for column in range(size)
    }
    for site in range(operator.nsites):
        state_by = defaultdict(list)
        for index, state in enumerate(states):
            for (ql, physical, qr), block in state.site_blocks[site].items():
                state_by[(index, ql, physical)].append((qr, block))
        op_by = defaultdict(list)
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        contributions = []
        for (
            row,
            column,
            qbra,
            left_a,
            left_b,
            qket,
        ), environment in environments.items():
            for entry_a in op_by.get(left_a, ()):
                for entry_b in op_by.get(left_b, ()):
                    if entry_a.physical_out != entry_b.physical_out:
                        continue
                    for qbra_r, bra_block in state_by.get(
                        (row, qbra, entry_a.physical_in), ()
                    ):
                        for qket_r, ket_block in state_by.get(
                            (column, qket, entry_b.physical_in), ()
                        ):
                            contributions.append(
                                (
                                    (
                                        row,
                                        column,
                                        qbra_r,
                                        entry_a.right,
                                        entry_b.right,
                                        qket_r,
                                    ),
                                    entry_a.value.conjugate() * entry_b.value,
                                    bra_block,
                                    environment,
                                    ket_block,
                                )
                            )
        environments = _batched_accumulate(contributions)
    result = np.empty((size, size), dtype=np.complex128)
    total_qn = states[0].total_qn
    for row in range(size):
        for column in range(size):
            final = environments.get((row, column, total_qn, 0, 0, total_qn))
            result[row, column] = 0.0j if final is None else _scalar_complex(final[0, 0])
    return 0.5 * (result + result.conj().T)


def _pack_state_sites(states: Sequence[NativeMPS]):
    """Pack the explicit state index inside each allowed charge block.

    Padding occurs only within an allowed ``(q_left, physical, q_right)``
    block. No symmetry-forbidden tensor block is allocated.
    """

    maximum_bond_dims = []
    for bond in range(states[0].nsites + 1):
        qns = {
            qn for state in states for qn in state.bond_dims[bond]
        }
        maximum_bond_dims.append(
            {
                qn: max(state.bond_dims[bond].get(qn, 0) for state in states)
                for qn in qns
            }
        )

    packed_sites = []
    for site in range(states[0].nsites):
        keys = sorted(
            {
                key
                for state in states
                for key in state.site_blocks[site]
            }
        )
        packed = {}
        for key in keys:
            blocks = [state.site_blocks[site].get(key) for state in states]
            left = maximum_bond_dims[site][key[0]]
            right = maximum_bond_dims[site + 1][key[2]]
            data = xp.zeros(
                (len(states), left, right), dtype=xp.complex128
            )
            for index, block in enumerate(blocks):
                if block is not None:
                    data[index, : block.shape[0], : block.shape[1]] = block
            packed[key] = data
        packed_sites.append(packed)
    return packed_sites


def _packed_triple(left, environment, right):
    """All state-pair ``left^H environment right`` contractions."""

    temporary = xp.matmul(
        left.conj().transpose(0, 2, 1)[:, None, :, :],
        environment,
    )
    return xp.matmul(temporary, right[None, :, :, :])


def _packed_overlap_matrix(left_states, right_states, left_packed=None, right_packed=None):
    if not left_states or not right_states:
        return np.empty((len(left_states), len(right_states)), dtype=np.complex128)
    left_packed = left_packed or _pack_state_sites(left_states)
    right_packed = right_packed or _pack_state_sites(right_states)
    environments = {
        (0, 0): xp.ones(
            (len(left_states), len(right_states), 1, 1),
            dtype=xp.complex128,
        )
    }
    for site in range(left_states[0].nsites):
        left_by = {
            (key[0], key[1]): (key[2], block)
            for key, block in left_packed[site].items()
        }
        right_by = {
            (key[0], key[1]): (key[2], block)
            for key, block in right_packed[site].items()
        }
        pending = {}
        for ql, environment in environments.items():
            for physical in range(4):
                left_item = left_by.get((ql, physical))
                right_item = right_by.get((ql, physical))
                if left_item is None or right_item is None:
                    continue
                qr_left, left = left_item
                qr_right, right = right_item
                if qr_left != qr_right:
                    continue
                contribution = _packed_triple(left, environment, right)
                pending[qr_left] = pending.get(qr_left, 0.0) + contribution
        environments = pending
    final = environments.get(left_states[0].total_qn)
    if final is None:
        return np.zeros((len(left_states), len(right_states)), dtype=np.complex128)
    return np.asarray(asnumpy(final[:, :, 0, 0]), dtype=np.complex128)


def _packed_action_matrix(states, operator, packed_sites):
    size = len(states)
    environments = {
        ((0, 0), 0, (0, 0)): xp.ones(
            (size, size, 1, 1), dtype=xp.complex128
        )
    }
    for site in range(operator.nsites):
        state_by = {
            (key[0], key[1]): (key[2], block)
            for key, block in packed_sites[site].items()
        }
        op_by = defaultdict(list)
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        pending = {}
        for (qbra, left_mpo, qket), environment in environments.items():
            for entry in op_by.get(left_mpo, ()):
                bra_item = state_by.get((qbra, entry.physical_out))
                ket_item = state_by.get((qket, entry.physical_in))
                if bra_item is None or ket_item is None:
                    continue
                qbra_r, bra = bra_item
                qket_r, ket = ket_item
                key = (qbra_r, entry.right, qket_r)
                contribution = entry.value * _packed_triple(
                    bra, environment, ket
                )
                pending[key] = pending.get(key, 0.0) + contribution
        environments = pending
    final = environments.get((states[0].total_qn, 0, states[0].total_qn))
    if final is None:
        return np.zeros((size, size), dtype=np.complex128)
    return np.asarray(asnumpy(final[:, :, 0, 0]), dtype=np.complex128)


def _packed_double_action_matrix(states, operator, packed_sites):
    size = len(states)
    environments = {
        ((0, 0), 0, 0, (0, 0)): xp.ones(
            (size, size, 1, 1), dtype=xp.complex128
        )
    }
    for site in range(operator.nsites):
        state_by = {
            (key[0], key[1]): (key[2], block)
            for key, block in packed_sites[site].items()
        }
        op_by = defaultdict(list)
        for entry in operator.entries[site]:
            op_by[entry.left].append(entry)
        pending = {}
        for (qbra, left_a, left_b, qket), environment in environments.items():
            for entry_a in op_by.get(left_a, ()):
                bra_item = state_by.get((qbra, entry_a.physical_in))
                if bra_item is None:
                    continue
                for entry_b in op_by.get(left_b, ()):
                    if entry_a.physical_out != entry_b.physical_out:
                        continue
                    ket_item = state_by.get((qket, entry_b.physical_in))
                    if ket_item is None:
                        continue
                    qbra_r, bra = bra_item
                    qket_r, ket = ket_item
                    key = (
                        qbra_r,
                        entry_a.right,
                        entry_b.right,
                        qket_r,
                    )
                    factor = entry_a.value.conjugate() * entry_b.value
                    contribution = factor * _packed_triple(
                        bra, environment, ket
                    )
                    pending[key] = pending.get(key, 0.0) + contribution
        environments = pending
    final = environments.get(
        (states[0].total_qn, 0, 0, states[0].total_qn)
    )
    if final is None:
        return np.zeros((size, size), dtype=np.complex128)
    result = np.asarray(asnumpy(final[:, :, 0, 0]), dtype=np.complex128)
    return 0.5 * (result + result.conj().T)


def exact_mpo_apply(operator: NativeMPO, state: NativeMPS) -> NativeMPS:
    """Form the exact MPO--MPS product using charge-labelled direct products."""

    if operator.nsites != state.nsites:
        raise ValueError("MPO/MPS site mismatch")
    component_maps = []
    output_dims = []
    for bond, state_dims in enumerate(state.bond_dims):
        grouped: dict[Qn, list[tuple[int, Qn, int]]] = defaultdict(list)
        for mpo_index, mpo_qn in enumerate(operator.bond_qns[bond]):
            for state_qn, dimension in state_dims.items():
                grouped[add_qn(state_qn, mpo_qn)].append(
                    (mpo_index, state_qn, dimension)
                )
        maps = {}
        dims = {}
        for output_qn, pieces in sorted(grouped.items()):
            offset = 0
            for mpo_index, state_qn, dimension in pieces:
                maps[(mpo_index, state_qn)] = (
                    output_qn,
                    slice(offset, offset + dimension),
                )
                offset += dimension
            dims[output_qn] = offset
        component_maps.append(maps)
        output_dims.append(dims)

    site_blocks = []
    for site in range(state.nsites):
        blocks: dict[BlockKey, object] = {}
        state_by_physical = defaultdict(list)
        for key, value in state.site_blocks[site].items():
            state_by_physical[key[1]].append((key[0], key[2], value))
        for entry in operator.entries[site]:
            for qin_l, qin_r, input_block in state_by_physical.get(
                entry.physical_in, ()
            ):
                qout_l, left_slice = component_maps[site][(entry.left, qin_l)]
                qout_r, right_slice = component_maps[site + 1][
                    (entry.right, qin_r)
                ]
                key = (qout_l, entry.physical_out, qout_r)
                if key not in blocks:
                    blocks[key] = xp.zeros(
                        (output_dims[site][qout_l], output_dims[site + 1][qout_r]),
                        dtype=xp.complex128,
                    )
                blocks[key][left_slice, right_slice] += entry.value * input_block
        site_blocks.append(blocks)
    return NativeMPS(site_blocks, output_dims, state.total_qn)


def exact_linear_combination(
    states: Sequence[NativeMPS], coefficients: Sequence[complex]
) -> NativeMPS:
    """Construct one exact direct-sum MPS for a linear combination."""

    selected = [
        (state, complex(coefficient))
        for state, coefficient in zip(states, coefficients)
        if complex(coefficient) != 0.0j
    ]
    if not selected:
        raise ValueError("linear combination has no nonzero term")
    nsites = selected[0][0].nsites
    total_qn = selected[0][0].total_qn
    if any(state.nsites != nsites or state.total_qn != total_qn for state, _ in selected):
        raise ValueError("incompatible MPS linear combination")

    maps = []
    combined_dims = []
    for bond in range(nsites + 1):
        if bond in (0, nsites):
            dims = {(0, 0): 1} if bond == 0 else {total_qn: 1}
            maps.append(
                [
                    {qn: slice(0, dimension) for qn, dimension in state.bond_dims[bond].items()}
                    for state, _ in selected
                ]
            )
            combined_dims.append(dims)
            continue
        dims: dict[Qn, int] = defaultdict(int)
        bond_maps = []
        for state, _ in selected:
            state_map = {}
            for qn, dimension in state.bond_dims[bond].items():
                start = dims[qn]
                state_map[qn] = slice(start, start + dimension)
                dims[qn] += dimension
            bond_maps.append(state_map)
        maps.append(bond_maps)
        combined_dims.append(dict(dims))

    site_blocks = []
    for site in range(nsites):
        blocks: dict[BlockKey, object] = {}
        for term, (state, coefficient) in enumerate(selected):
            factor = coefficient if site == 0 else 1.0 + 0.0j
            for key, source in state.site_blocks[site].items():
                ql, physical, qr = key
                if key not in blocks:
                    blocks[key] = xp.zeros(
                        (combined_dims[site][ql], combined_dims[site + 1][qr]),
                        dtype=xp.complex128,
                    )
                blocks[key][maps[site][term][ql], maps[site + 1][term][qr]] += (
                    factor * source
                )
        site_blocks.append(blocks)
    return NativeMPS(site_blocks, combined_dims, total_qn)


def _src_decomposition(
    matrix,
    candidate_rank: int,
    *,
    power_steps: int,
    oversampling: int,
    rng: np.random.Generator,
    real_input: bool,
):
    """Return randomized left vectors, transfers, and estimated singular values."""

    rows, columns = matrix.shape
    if real_input:
        matrix = xp.asarray(xp.real(matrix), dtype=xp.float64)
    capacity = min(rows, columns)
    sketch_rank = min(capacity, max(candidate_rank, oversampling))
    if sketch_rank == capacity:
        basis, _ = xp.linalg.qr(matrix, mode="reduced")
    else:
        if real_input:
            omega = asxp(rng.standard_normal((columns, sketch_rank)))
        else:
            omega = asxp(
                rng.standard_normal((columns, sketch_rank))
                + 1j * rng.standard_normal((columns, sketch_rank))
            ) / math.sqrt(2.0)
        basis, _ = xp.linalg.qr(matrix @ omega, mode="reduced")
        for _ in range(power_steps):
            dual, _ = xp.linalg.qr(matrix.conj().T @ basis, mode="reduced")
            basis, _ = xp.linalg.qr(matrix @ dual, mode="reduced")
    projected = basis.conj().T @ matrix
    gram = projected @ projected.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    values, vectors = xp.linalg.eigh(gram)
    order = xp.argsort(values)[::-1]
    basis = basis @ vectors[:, order]
    transfer = basis.conj().T @ matrix
    singular_values = xp.sqrt(xp.maximum(values[order], 0.0))
    return basis, transfer, singular_values


def _src_decompositions_grouped(
    matrices: Sequence[object],
    candidate_rank: int,
    *,
    power_steps: int,
    oversampling: int,
    seeds: Sequence[int],
    real_input: bool,
):
    """Batched SRC kernels for charge blocks sharing an exact matrix shape."""

    prepared = [
        xp.asarray(xp.real(matrix), dtype=xp.float64)
        if real_input
        else matrix
        for matrix in matrices
    ]
    if os.environ.get("GPLHR_PADDED_SRC", "0").lower() in {
        "1", "true", "yes", "on"
    }:
        return _src_decompositions_padded(
            prepared,
            candidate_rank,
            power_steps=power_steps,
            oversampling=oversampling,
            seeds=seeds,
            real_input=real_input,
        )
    outputs = [None] * len(prepared)
    groups = defaultdict(list)
    for index, matrix in enumerate(prepared):
        capacity = min(matrix.shape)
        sketch_rank = min(capacity, max(candidate_rank, oversampling))
        groups[(matrix.shape, matrix.dtype.str, sketch_rank)].append(index)

    for (_, _, sketch_rank), indices in groups.items():
        batch = xp.stack([prepared[index] for index in indices])
        rows, columns = batch.shape[1:]
        capacity = min(rows, columns)
        if sketch_rank == capacity:
            basis, _ = xp.linalg.qr(batch, mode="reduced")
        else:
            omega_blocks = []
            for index in indices:
                rng = np.random.default_rng(seeds[index])
                if real_input:
                    omega = rng.standard_normal((columns, sketch_rank))
                else:
                    omega = (
                        rng.standard_normal((columns, sketch_rank))
                        + 1j * rng.standard_normal((columns, sketch_rank))
                    ) / math.sqrt(2.0)
                omega_blocks.append(asxp(omega))
            omega_batch = xp.stack(omega_blocks)
            basis, _ = xp.linalg.qr(batch @ omega_batch, mode="reduced")
            for _ in range(power_steps):
                dual, _ = xp.linalg.qr(
                    batch.conj().transpose(0, 2, 1) @ basis,
                    mode="reduced",
                )
                basis, _ = xp.linalg.qr(batch @ dual, mode="reduced")
        projected = basis.conj().transpose(0, 2, 1) @ batch
        gram = projected @ projected.conj().transpose(0, 2, 1)
        gram = 0.5 * (gram + gram.conj().transpose(0, 2, 1))
        values, vectors = xp.linalg.eigh(gram)
        order = xp.argsort(values, axis=1)[:, ::-1]
        values = xp.take_along_axis(values, order, axis=1)
        vectors = xp.take_along_axis(vectors, order[:, None, :], axis=2)
        basis = basis @ vectors
        transfer = basis.conj().transpose(0, 2, 1) @ batch
        singular_values = xp.sqrt(xp.maximum(values, 0.0))
        _BACKEND_COUNTERS["batched_src_groups"] += 1
        _BACKEND_COUNTERS["batched_src_blocks"] += len(indices)
        for batch_index, output_index in enumerate(indices):
            outputs[output_index] = (
                basis[batch_index],
                transfer[batch_index],
                singular_values[batch_index],
            )
    return outputs


def _src_decompositions_padded(
    matrices: Sequence[object],
    candidate_rank: int,
    *,
    power_steps: int,
    oversampling: int,
    seeds: Sequence[int],
    real_input: bool,
):
    """Fuse variable-size allowed charge blocks with temporary zero padding."""

    outputs = [None] * len(matrices)
    sketch_ranks = [
        min(min(matrix.shape), max(candidate_rank, oversampling))
        for matrix in matrices
    ]
    subsets = defaultdict(list)
    for index, (matrix, sketch_rank) in enumerate(zip(matrices, sketch_ranks)):
        subsets[sketch_rank < min(matrix.shape)].append(index)

    for randomized, indices in subsets.items():
        maximum_rows = max(matrices[index].shape[0] for index in indices)
        maximum_columns = max(matrices[index].shape[1] for index in indices)
        maximum_sketch = max(sketch_ranks[index] for index in indices)
        matrix_batch = xp.zeros(
            (len(indices), maximum_rows, maximum_columns),
            dtype=matrices[indices[0]].dtype,
        )
        sample_batch = xp.zeros(
            (len(indices), maximum_rows, maximum_sketch),
            dtype=matrices[indices[0]].dtype,
        )
        for batch_index, index in enumerate(indices):
            matrix = matrices[index]
            rows, columns = matrix.shape
            sketch_rank = sketch_ranks[index]
            matrix_batch[batch_index, :rows, :columns] = matrix
            if randomized:
                rng = np.random.default_rng(seeds[index])
                if real_input:
                    omega = rng.standard_normal((columns, sketch_rank))
                else:
                    omega = (
                        rng.standard_normal((columns, sketch_rank))
                        + 1j * rng.standard_normal((columns, sketch_rank))
                    ) / math.sqrt(2.0)
                sample_batch[batch_index, :rows, :sketch_rank] = (
                    matrix @ asxp(omega)
                )
            elif rows >= columns:
                sample_batch[batch_index, :rows, :sketch_rank] = matrix
            else:
                sample_batch[batch_index, :rows, :sketch_rank] = xp.eye(
                    rows, dtype=matrix.dtype
                )

        basis, _ = xp.linalg.qr(sample_batch, mode="reduced")

        def mask_columns(batch):
            for batch_index, index in enumerate(indices):
                rows = matrices[index].shape[0]
                sketch_rank = sketch_ranks[index]
                if rows < batch.shape[1]:
                    batch[batch_index, rows:, :] = 0
                if sketch_rank < batch.shape[2]:
                    batch[batch_index, :, sketch_rank:] = 0
            return batch

        basis = mask_columns(basis)
        if randomized:
            for _ in range(power_steps):
                dual = matrix_batch.conj().transpose(0, 2, 1) @ basis
                dual, _ = xp.linalg.qr(dual, mode="reduced")
                for batch_index, index in enumerate(indices):
                    columns = matrices[index].shape[1]
                    sketch_rank = sketch_ranks[index]
                    if columns < dual.shape[1]:
                        dual[batch_index, columns:, :] = 0
                    if sketch_rank < dual.shape[2]:
                        dual[batch_index, :, sketch_rank:] = 0
                basis, _ = xp.linalg.qr(matrix_batch @ dual, mode="reduced")
                basis = mask_columns(basis)

        projected = basis.conj().transpose(0, 2, 1) @ matrix_batch
        gram = projected @ projected.conj().transpose(0, 2, 1)
        gram = 0.5 * (gram + gram.conj().transpose(0, 2, 1))
        values, vectors = xp.linalg.eigh(gram)
        order = xp.argsort(values, axis=1)[:, ::-1]
        values = xp.take_along_axis(values, order, axis=1)
        vectors = xp.take_along_axis(vectors, order[:, None, :], axis=2)
        rotated_basis = basis @ vectors
        transfer = rotated_basis.conj().transpose(0, 2, 1) @ matrix_batch
        singular_values = xp.sqrt(xp.maximum(values, 0.0))
        _BACKEND_COUNTERS["padded_src_launches"] += 1
        _BACKEND_COUNTERS["padded_src_blocks"] += len(indices)
        for batch_index, index in enumerate(indices):
            rows, columns = matrices[index].shape
            sketch_rank = sketch_ranks[index]
            outputs[index] = (
                rotated_basis[batch_index, :rows, :sketch_rank],
                transfer[batch_index, :sketch_rank, :columns],
                singular_values[batch_index, :sketch_rank],
            )
    return outputs


def right_canonicalize_exact(
    state: NativeMPS,
    *,
    numerical_rank_tolerance: float | None = None,
    consume_input: bool = False,
) -> NativeMPS:
    """Put an MPS in right-canonical form, charge block by charge block.

    Direct-sum additions and MPO products contain large gauge-redundant bond
    spaces.  A left-to-right truncation is meaningful only after the untouched
    right block is isometric.  This sweep removes numerical null directions
    but performs no user-controlled lossy truncation.
    """

   
    result = state if consume_input else state.copy()
    real_input = _arrays_are_numerically_real(
        block
        for blocks in result.site_blocks
        for block in blocks.values()
    )
    for site in range(result.nsites - 1, 0, -1):
        keys_by_left = defaultdict(list)
        for key in result.site_blocks[site]:
            keys_by_left[key[0]].append(key)
        new_left_dims: dict[Qn, int] = {}
        new_site: dict[BlockKey, object] = {}
        transfers: dict[Qn, object] = {}
        records = []
        for left_qn, keys in sorted(keys_by_left.items()):
            keys = sorted(keys)
            pieces = [result.site_blocks[site][key] for key in keys]
            matrix = xp.concatenate(pieces, axis=1)
            if real_input:
                matrix = xp.asarray(xp.real(matrix), dtype=xp.float64)
            
            records.append((left_qn, keys, pieces, matrix))

        factorizations = _grouped_right_factorization(
            [record[-1] for record in records]
        )
        for (left_qn, keys, pieces, matrix), (right, transfer) in zip(
            records, factorizations
        ):
            rank = int(right.shape[0])
            if rank == 0:
                continue
            new_left_dims[left_qn] = rank
            transfers[left_qn] = xp.ascontiguousarray(transfer)
            if USE_GPU and os.environ.get("GPLHR_EXACT_GAUGE", "qr").lower() == "qr":
                _BACKEND_COUNTERS["gpu_exact_unpivoted_qr_calls"] += 1
            offset = 0
            for key, piece in zip(keys, pieces):
                columns = piece.shape[1]
                new_site[key] = xp.ascontiguousarray(
                    right[:rank, offset : offset + columns]
                )
                offset += columns

        previous_site = {}
        for (left_qn, physical, right_qn), block in result.site_blocks[
            site - 1
        ].items():
            transfer = transfers.get(right_qn)
            if transfer is None:
                continue
            previous_site[(left_qn, physical, right_qn)] = xp.ascontiguousarray(
                block @ transfer
            )
        result.site_blocks[site - 1] = previous_site
        result.site_blocks[site] = new_site
        result.bond_dims[site] = new_left_dims
    result.validate()
    result.metadata["right_canonical"] = True
    result.metadata["numerically_real_blocks"] = real_input
    return result


def _src_round_two_sweep(
    state: NativeMPS,
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
    oversampling_factor: float = 1.5,
    additive_oversampling: int = 10,
    consume_input: bool = False,
) -> tuple[NativeMPS, dict]:
    """Perform one native, charge-by-charge left-to-right SRC sweep.

    The returned state has the scale of ``state``.  In particular, an MPO
    application must not be normalized here: GPLHR uses the magnitude and
    phase of ``H |psi>`` in its projected matrices and residuals.
    """

    if max_bond < 1:
        raise ValueError("max_bond must be positive")

    started = time.perf_counter()
    canonical_started = time.perf_counter()
    result = right_canonicalize_exact(state, consume_input=consume_input)
    real_input = bool(result.metadata.get("numerically_real_blocks", False))
    canonical_seconds = time.perf_counter() - canonical_started
    sweep_started = time.perf_counter()
    maximum_discarded = 0.0
    sum_discarded = 0.0
    capped_bonds = 0
    for site in range(result.nsites - 1):
        new_right_dims: dict[Qn, int] = {}
        new_site: dict[BlockKey, object] = {}
        transfers: dict[Qn, object] = {}
        keys_by_right = defaultdict(list)
        for key in result.site_blocks[site]:
            keys_by_right[key[2]].append(key)
        decomposed = {}
        sketch = max(
            int(math.ceil(oversampling_factor * max_bond)),
            max_bond + additive_oversampling,
        )
        records = []
        for block_index, (right_qn, keys) in enumerate(sorted(keys_by_right.items())):
            keys = sorted(keys)
            pieces = [result.site_blocks[site][key] for key in keys]
            matrix = xp.concatenate(pieces, axis=0)
            records.append(
                (
                    right_qn,
                    keys,
                    pieces,
                    matrix,
                    seed + 1009 * site + 37 * block_index,
                )
            )
        outputs = _src_decompositions_grouped(
            [record[3] for record in records],
            max_bond,
            power_steps=power_steps,
            oversampling=sketch,
            seeds=[record[4] for record in records],
            real_input=real_input,
        )
        for (right_qn, keys, pieces, matrix, _), (
            basis,
            transfer,
            singular_values,
        ) in zip(records, outputs):
            decomposed[right_qn] = (
                keys,
                pieces,
                matrix,
                basis,
                transfer,
                singular_values,
            )

       
        candidates = []
        ordered_qns = list(decomposed)
        singular_lengths = [decomposed[qn][-1].size for qn in ordered_qns]
        singular_flat = (
            xp.concatenate([decomposed[qn][-1] for qn in ordered_qns])
            if ordered_qns
            else xp.empty(0, dtype=xp.float64)
        )
        total_energy_device = xp.sum(
            xp.stack(
                [
                    xp.sum(xp.abs(decomposition[2]) ** 2)
                    for decomposition in decomposed.values()
                ]
            )
        )
        
        selection_payload = xp.concatenate(
            [singular_flat, total_energy_device.reshape(1)]
        )
        selection_cpu = np.asarray(
            asnumpy(selection_payload), dtype=np.float64
        )
        singular_cpu = selection_cpu[:-1]
        total_energy = float(selection_cpu[-1])
        offset = 0
        for right_qn, length in zip(ordered_qns, singular_lengths):
            candidates.extend(
                (float(value), right_qn)
                for value in singular_cpu[offset : offset + length]
            )
            offset += length
        candidates.sort(key=lambda item: item[0], reverse=True)
        retained_by_qn = {right_qn: 0 for right_qn in decomposed}
        for _, right_qn in candidates[:max_bond]:
            retained_by_qn[right_qn] += 1

        retained_energy = 0.0
        offset = 0
        for right_qn, length in zip(ordered_qns, singular_lengths):
            retained = retained_by_qn.get(right_qn, 0)
            retained_energy += float(
                np.sum(singular_cpu[offset : offset + retained] ** 2)
            )
            offset += length
        discarded = max(
            0.0, 1.0 - retained_energy / max(total_energy, 1.0e-300)
        )
        maximum_discarded = max(maximum_discarded, discarded)
        sum_discarded += discarded

        for right_qn, (
            keys,
            pieces,
            matrix,
            basis,
            transfer,
            singular_values,
        ) in decomposed.items():
            retained = retained_by_qn.get(right_qn, 0)
            if retained == 0:
                capped_bonds += int(singular_values.size > 0)
                continue
            basis = basis[:, :retained]
            transfer = transfer[:retained]
            new_right_dims[right_qn] = retained
            transfers[right_qn] = transfer
            capped_bonds += int(singular_values.size > retained)
            offset = 0
            for key, piece in zip(keys, pieces):
                rows = piece.shape[0]
                new_site[key] = xp.ascontiguousarray(basis[offset : offset + rows])
                offset += rows
        next_site = {}
        for (left_qn, physical, right_qn), block in result.site_blocks[site + 1].items():
            
            if left_qn not in transfers:
                continue
            next_site[(left_qn, physical, right_qn)] = xp.ascontiguousarray(
                transfers[left_qn] @ block
            )
        result.site_blocks[site] = new_site
        result.site_blocks[site + 1] = next_site
        result.bond_dims[site + 1] = new_right_dims
    result.validate(max_bond=max_bond)
    info = {
        "method": "native U(1)xU(1) successive randomized compression",
        "max_bond": int(max_bond),
        "max_bond_convention": "maximum sector-summed virtual-bond dimension per MPS",
        "block_bond_dimensions": [max(d.values()) for d in result.bond_dims],
        "bond_dimensions": result.bond_dimensions,
        "maximum_charge_block_bond": result.max_charge_block_bond,
        "maximum_sector_summed_bond": result.max_total_bond,
        "maximum_discarded_weight_estimate": maximum_discarded,
        "sum_discarded_weight_estimate": sum_discarded,
        "capped_charge_blocks": capped_bonds,
        "stored_elements": result.stored_elements,
        "power_steps": power_steps,
        "input_ownership": "consumed" if consume_input else "copied",
        "right_canonicalization_seconds": canonical_seconds,
        "compression_sweep_seconds": time.perf_counter() - sweep_started,
        "total_seconds": time.perf_counter() - started,
    }
    result.metadata["last_src"] = info
    return result, info


def src_round(
    state: NativeMPS,
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
    oversampling_factor: float = 1.5,
    additive_oversampling: int = 10,
    consume_input: bool = False,
) -> tuple[NativeMPS, dict]:
    """Use exact right canonicalization followed by the validated SRC sweep.

    ``consume_input`` is an explicit ownership transfer for temporary exact
    MPO products and direct sums.  It changes only allocation lifetime: the
    factorization, random seeds, truncation policy, and returned tensors are
    identical to the default non-mutating path.
    """

    if max_bond < 1:
        raise ValueError("max_bond must be positive")
    return _src_round_two_sweep(
        state,
        max_bond,
        seed=seed,
        power_steps=power_steps,
        oversampling_factor=oversampling_factor,
        additive_oversampling=additive_oversampling,
        consume_input=consume_input,
    )

    


def src_sum(
    states: Sequence[NativeMPS],
    coefficients: Sequence[complex],
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
) -> tuple[NativeMPS, dict]:
    started = time.perf_counter()
    sum_started = time.perf_counter()
    exact = exact_linear_combination(states, coefficients)
    exact_sum_seconds = time.perf_counter() - sum_started
    result, report = src_round(
        exact,
        max_bond,
        seed=seed,
        power_steps=power_steps,
        consume_input=True,
    )
    report["operation"] = "mps_linear_combination"
    report["exact_direct_sum_seconds"] = exact_sum_seconds
    report["total_sum_and_src_seconds"] = time.perf_counter() - started
    result.metadata["last_src"] = report
    return result, report


def src_apply(
    operator: NativeMPO,
    state: NativeMPS,
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
    oversampling_factor: float = 1.5,
    additive_oversampling: int = 10,
) -> tuple[NativeMPS, dict]:
    """Apply an MPO and compress the enlarged bonds with SRC.

    The output is deliberately *not* normalized.
    """

    exact = exact_mpo_apply(operator, state)
    result, info = src_round(
        exact,
        max_bond,
        seed=seed,
        power_steps=power_steps,
        oversampling_factor=oversampling_factor,
        additive_oversampling=additive_oversampling,
        consume_input=True,
    )
    info["operation"] = "mpo_mps_action"
    result.metadata["last_src"] = info
    return result, info


def random_mps(
    sites: int,
    total_qn: Qn,
    max_bond: int,
    *,
    seed: int,
) -> NativeMPS:
    """Create a real random symmetry-native MPS without configuration vectors.

    The Hubbard MPO and target used by the benchmark are real.  A complex
    random start needlessly breaks that invariant arithmetic subspace and can
    create complex projected Ritz values before the first restart.
    """

    nup, ndown = total_qn
    bond_dims: list[dict[Qn, int]] = []
    for cut in range(sites + 1):
        right = sites - cut
        capacities = {}
        for up_left in range(cut + 1):
            up_right = nup - up_left
            if not 0 <= up_right <= right:
                continue
            for down_left in range(cut + 1):
                down_right = ndown - down_left
                if not 0 <= down_right <= right:
                    continue
                left_count = math.comb(cut, up_left) * math.comb(cut, down_left)
                right_count = math.comb(right, up_right) * math.comb(right, down_right)
                capacities[(up_left, down_left)] = min(left_count, right_count)
        ordered = sorted(capacities, key=lambda qn: (-capacities[qn], qn))
        dims: dict[Qn, int] = {}
        remaining = min(max_bond, sum(capacities.values()))
        while remaining:
            progressed = False
            for qn in ordered:
                if dims.get(qn, 0) >= capacities[qn]:
                    continue
                dims[qn] = dims.get(qn, 0) + 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
            if not progressed:
                break
        bond_dims.append(dims)
    rng = np.random.default_rng(seed)
    site_blocks = []
    for site in range(sites):
        blocks = {}
        for left_qn, left_dim in bond_dims[site].items():
            for physical, local_qn in enumerate(LOCAL_QN):
                right_qn = add_qn(left_qn, local_qn)
                right_dim = bond_dims[site + 1].get(right_qn)
                if right_dim is None:
                    continue
                blocks[(left_qn, physical, right_qn)] = asxp(
                    rng.standard_normal((left_dim, right_dim))
                    / math.sqrt(max(left_dim * 4, 1))
                )
        site_blocks.append(blocks)
    state = NativeMPS(site_blocks, bond_dims, total_qn)
    rounded, _ = src_round(
        state,
        max_bond,
        seed=seed + 7919,
        consume_input=True,
    )
    return normalize(rounded)


def maximum_tensor_imaginary_element(state: NativeMPS) -> float:
    """Largest imaginary entry in the stored MPS gauge."""

    values = [
        xp.max(xp.abs(xp.imag(block)))
        for site in state.site_blocks
        for block in site.values()
    ]
    return 0.0 if not values else _scalar_float(xp.max(xp.stack(values)))


def relative_tensor_imaginary_norm(state: NativeMPS) -> float:
    """Gauge-dependent diagnostic for preservation of real arithmetic."""

    total_terms = []
    imaginary_terms = []
    for site in state.site_blocks:
        for block in site.values():
            total_terms.append(xp.real(xp.vdot(block, block)))
            imaginary_terms.append(
                xp.real(xp.vdot(xp.imag(block), xp.imag(block)))
            )
    total, imaginary = np.asarray(
        asnumpy(
            xp.stack(
                [
                    xp.sum(xp.stack(total_terms)),
                    xp.sum(xp.stack(imaginary_terms)),
                ]
            )
        ),
        dtype=np.float64,
    )
    return math.sqrt(imaginary / max(total, np.finfo(np.float64).tiny))


def tensor_block_imaginary_diagnostics(
    states: Sequence[NativeMPS],
) -> tuple[float, float]:
    """Return both block diagnostics with one GPU-to-CPU synchronization."""

    if not states:
        return 0.0, 0.0
    rows = []
    for state in states:
        blocks = [block for site in state.site_blocks for block in site.values()]
        rows.append(
            xp.stack(
                [
                    xp.max(
                        xp.stack(
                            [xp.max(xp.abs(xp.imag(block))) for block in blocks]
                        )
                    ),
                    xp.sum(
                        xp.stack([xp.real(xp.vdot(block, block)) for block in blocks])
                    ),
                    xp.sum(
                        xp.stack(
                            [
                                xp.real(xp.vdot(xp.imag(block), xp.imag(block)))
                                for block in blocks
                            ]
                        )
                    ),
                ]
            )
        )
    diagnostics = np.asarray(asnumpy(xp.stack(rows)), dtype=np.float64)
    maximum_element = float(np.max(diagnostics[:, 0]))
    relative = np.sqrt(
        diagnostics[:, 2]
        / np.maximum(diagnostics[:, 1], np.finfo(np.float64).tiny)
    )
    return maximum_element, float(np.max(relative))


def product_mps(local_states: Sequence[int]) -> NativeMPS:
    """Build an exact charge-labelled occupation product state."""

    if not local_states:
        raise ValueError("a product MPS needs at least one site")
    charge = (0, 0)
    bond_dims: list[dict[Qn, int]] = [{charge: 1}]
    site_blocks: list[dict[BlockKey, object]] = []
    for physical in local_states:
        physical = int(physical)
        if not 0 <= physical < len(LOCAL_QN):
            raise ValueError("invalid local product-state index")
        next_charge = add_qn(charge, LOCAL_QN[physical])
        site_blocks.append(
            {
                (charge, physical, next_charge): xp.ones(
                    (1, 1), dtype=xp.complex128
                )
            }
        )
        charge = next_charge
        bond_dims.append({charge: 1})
    return NativeMPS(site_blocks, bond_dims, charge)


def diagonal_product_apply(
    state: NativeMPS,
    local_diagonals: Sequence[Sequence[complex]],
) -> NativeMPS:
    """Apply an exact bond-one diagonal product operator to an MPS."""

    if len(local_diagonals) != state.nsites:
        raise ValueError("diagonal product and MPS have different site counts")
    result = state.copy()
    for site, diagonal in enumerate(local_diagonals):
        diagonal = np.asarray(diagonal, dtype=np.complex128)
        if diagonal.shape != (len(LOCAL_QN),):
            raise ValueError("each local diagonal must have four entries")
        for key in result.site_blocks[site]:
            result.site_blocks[site][key] *= diagonal[key[1]]
    result.validate()
    return result


def pack_state_sites(states: Sequence[NativeMPS]):
    """Prepare reusable GPU block packs for repeated contractions."""

    return _pack_state_sites(states) if USE_GPU and states else None


def cross_gram(
    left: Sequence[NativeMPS],
    right: Sequence[NativeMPS],
    *,
    left_packed=None,
    right_packed=None,
) -> np.ndarray:
    """Contract a rectangular block Gram matrix."""

    if USE_GPU:
        return _packed_overlap_matrix(
            left,
            right,
            left_packed=left_packed,
            right_packed=right_packed,
        )

    result = np.empty((len(left), len(right)), dtype=np.complex128)
    for row, bra in enumerate(left):
        for column, ket in enumerate(right):
            result[row, column] = overlap(bra, ket)
    return result


def rotate_src(
    states: Sequence[NativeMPS],
    coefficients: np.ndarray,
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
) -> tuple[list[NativeMPS], list[dict]]:
    """Materialize a small state-index rotation using one SRC fit per output."""

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(states):
        raise ValueError("state rotation has incompatible dimensions")
    def rotate_column(column):
        started = time.perf_counter()
        state, report = src_sum(
            states,
            coefficients[:, column],
            max_bond,
            seed=seed + 104729 * column,
            power_steps=power_steps,
        )
        report["rotation_output_seconds"] = time.perf_counter() - started
        return state, report

    rotated = gpu_parallel_map(rotate_column, range(coefficients.shape[1]))
    output = [item[0] for item in rotated]
    reports = [item[1] for item in rotated]
    return output, reports


def orthonormalize_src(
    states: Sequence[NativeMPS],
    max_bond: int,
    *,
    seed: int,
    power_steps: int = 2,
    rank_cutoff: float = 1.0e-12,
    passes: int = 2,
) -> tuple[list[NativeMPS], dict]:
    """Block modified Gram--Schmidt with one SRC sum per pass and vector."""

    if passes < 1:
        raise ValueError("orthogonalization passes must be positive")
    basis: list[NativeMPS] = []
    reports = []
    dropped = 0
    for source_index, source in enumerate(states):
        candidate = source.copy()
        initial_norm = norm(candidate)
        for orthogonalization_pass in range(passes):
            if basis:
                coefficients = [-overlap(vector, candidate) for vector in basis]
                candidate, report = src_sum(
                    [candidate, *basis],
                    [1.0, *coefficients],
                    max_bond,
                    seed=(
                        seed
                        + 104729 * source_index
                        + 1009 * orthogonalization_pass
                    ),
                    power_steps=power_steps,
                )
                reports.append(
                    {
                        "source": source_index,
                        "pass": orthogonalization_pass,
                        **report,
                    }
                )
        candidate_norm = norm(candidate)
        if candidate_norm <= rank_cutoff * max(initial_norm, 1.0e-300):
            dropped += 1
            continue
        basis.append(candidate.scale(1.0 / candidate_norm))
    gram = gram_matrix(basis)
    return basis, {
        "method": "two-pass MPS BMGS with charge-resolved SRC additions",
        "input_columns": len(states),
        "retained_columns": len(basis),
        "dropped_columns": dropped,
        "passes": passes,
        "maximum_orthogonality_error": float(
            np.linalg.norm(gram - np.eye(len(basis)), ord=np.inf)
        ),
        "src_additions": reports,
    }


def gram_matrix(states: Sequence[NativeMPS]) -> np.ndarray:
    if USE_GPU:
        packed = _pack_state_sites(states)
        result = _packed_overlap_matrix(
            states, states, left_packed=packed, right_packed=packed
        )
        return 0.5 * (result + result.conj().T)
    size = len(states)
    result = np.empty((size, size), dtype=np.complex128)
    for row in range(size):
        for column in range(row, size):
            value = overlap(states[row], states[column])
            result[row, column] = value
            result[column, row] = value.conjugate()
    return 0.5 * (result + result.conj().T)


def projected_matrices(
    states: Sequence[NativeMPS], operator: NativeMPO, *, include_l: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    size = len(states)
    if USE_GPU:
        packed = _pack_state_sites(states)
        gram = _packed_overlap_matrix(
            states, states, left_packed=packed, right_packed=packed
        )
        gram = 0.5 * (gram + gram.conj().T)
        projected = _packed_action_matrix(states, operator, packed)
        shifted_gram = (
            _packed_double_action_matrix(states, operator, packed)
            if include_l
            else None
        )
        return gram, projected, shifted_gram

    gram = gram_matrix(states)
    projected = np.empty((size, size), dtype=np.complex128)
    shifted_gram = (
        np.empty((size, size), dtype=np.complex128) if include_l else None
    )
    for row in range(size):
        for column in range(size):
            projected[row, column] = action_inner(
                states[row], operator, states[column]
            )
        if include_l:
            for column in range(row, size):
                value = double_action_inner(states[row], operator, states[column])
                shifted_gram[row, column] = value
                shifted_gram[column, row] = value.conjugate()
    if include_l:
        shifted_gram = 0.5 * (shifted_gram + shifted_gram.conj().T)
    return gram, projected, shifted_gram


def residual_norm_from_small(
    gram: np.ndarray,
    projected: np.ndarray,
    action_gram: np.ndarray,
    coefficients: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return norms of ``H Z c_j - theta_j Z c_j`` from contractions."""

    output = []
    for column, value in enumerate(values):
        vector = coefficients[:, column]
        squared = (
            np.vdot(vector, action_gram @ vector)
            - value * np.vdot(vector, projected.conj().T @ vector)
            - value.conjugate() * np.vdot(vector, projected @ vector)
            + abs(value) ** 2 * np.vdot(vector, gram @ vector)
        )
        output.append(math.sqrt(max(float(squared.real), 0.0)))
    return np.asarray(output)


__all__ = [
    "LOCAL_QN",
    "NativeMPO",
    "NativeMPS",
    "action_inner",
    "double_action_inner",
    "exact_linear_combination",
    "exact_mpo_apply",
    "cross_gram",
    "diagonal_similarity_mpo",
    "diagonal_product_apply",
    "gram_matrix",
    "gpu_parallel_map",
    "maximum_tensor_imaginary_element",
    "native_backend_info",
    "normalize",
    "norm",
    "overlap",
    "orthonormalize_src",
    "pack_state_sites",
    "projected_matrices",
    "product_mps",
    "random_mps",
    "relative_tensor_imaginary_norm",
    "residual_norm_from_small",
    "right_canonicalize_exact",
    "rotate_src",
    "src_apply",
    "src_round",
    "src_sum",
    "tensor_block_imaginary_diagnostics",
]
