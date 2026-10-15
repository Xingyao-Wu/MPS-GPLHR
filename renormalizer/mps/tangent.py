"""Fixed-rank MPS tangent vectors and projected MPO actions.

This module implements the one-site tangent-space construction used by the
TaSK algorithm of Kovalska, von Delft, and Gleis (arXiv:2510.07279).  A tangent
vector is kept as a list of local tensors; the exponentially large sum of MPS
wavefunctions is never formed during a Hamiltonian application.

Tangent coordinates use Renormalizer's quantum-number metadata.  Their public
representation follows the rest of the MPS package (dense arrays with exact
zeros outside the selected sector), while all masks, gauge constraints, and
dimension counts are charge-block aware.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import SimpleQueue
import time
from typing import Iterable, List, Optional

import numpy as np

from renormalizer.mps.lib import contract_one_site
from renormalizer.mps.matrix import asnumpy, multi_tensor_contract
from renormalizer.mps.svd_qn import add_outer, get_qn_mask


def _left_message(environment, bra, operator, ket):
    """Contract one site into a left three-leg environment.

    Environment axes are ``(bra bond, MPO bond, ket bond)`` and an MPO tensor
    has Renormalizer ordering ``(left, physical out, physical in, right)``.
    """

    return asnumpy(
        contract_one_site(
            environment, ket, operator, "L", ms_conj=bra.conj()
        )
    )


def _right_message(environment, bra, operator, ket):
    """Contract one site into a right three-leg environment."""

    return asnumpy(
        contract_one_site(
            environment, ket, operator, "R", ms_conj=bra.conj()
        )
    )


def _effective_one_site(left, operator, right, ket):
    """Leave the three bra legs open in a one-site network contraction."""

    path = [
        ([0, 1], "abc, cek -> abek"),
        ([2, 0], "abek, bdef -> akdf"),
        ([1, 0], "akdf, lfk -> adl"),
    ]
    return asnumpy(
        multi_tensor_contract(path, left, ket, operator, right)
    )


@dataclass
class MpsTangentVector:
    """A vector in the one-site tangent space of a fixed reference MPS.

    Instances should normally be created with :meth:`MpsTangentSpace.vector`,
    :meth:`MpsTangentSpace.zeros`, or :meth:`MpsTangentSpace.random`.
    """

    space: "MpsTangentSpace"
    tensors: List[np.ndarray]

    def copy(self) -> "MpsTangentVector":
        return type(self)(self.space, [tensor.copy() for tensor in self.tensors])

    def dot(self, other: "MpsTangentVector") -> complex:
        self.space._check_vector(other)
        return sum(
            (np.vdot(left, right) for left, right in zip(self.tensors, other.tensors)),
            0.0j,
        )

    @property
    def norm(self) -> float:
        value = self.dot(self).real
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        return float(np.sqrt(value))

    def scale(self, value, inplace: bool = False) -> "MpsTangentVector":
        result = self if inplace else self.copy()
        result.tensors = [value * tensor for tensor in result.tensors]
        return result

    def axpy(self, value, other: "MpsTangentVector") -> "MpsTangentVector":
        """Perform ``self += value * other`` and return ``self``."""

        self.space._check_vector(other)
        self.tensors = [
            tensor + value * increment
            for tensor, increment in zip(self.tensors, other.tensors)
        ]
        return self

    def restore_gauge(self, inplace: bool = False) -> "MpsTangentVector":
        """Reapply the frozen tangent gauge after cancellation-prone algebra."""

        result = self if inplace else self.copy()
        result.tensors = self.space.project_gauge(result.tensors)
        return result

    def normalize(self) -> "MpsTangentVector":
        self.restore_gauge(inplace=True)
        norm = self.norm
        if norm == 0.0:
            raise ValueError("cannot normalize a zero tangent vector")
        return self.scale(1.0 / norm, inplace=True)

    def to_dense(self) -> np.ndarray:
        """Materialize the tangent wavefunction (intended for diagnostics)."""

        return self.space.to_dense(self)


class MpsTangentSpace:
    r"""One-site tangent space orthogonal to a reference MPS.

    The stored left and right isometries implement Eq. (S3)--(S4) of
    arXiv:2510.07279.  If ``A[l]`` and ``B[l]`` are the two canonical forms, a
    vector is represented as

    .. math::

       |\Phi[T]\rangle = \sum_l A_1\cdots A_{l-1}T_lB_{l+1}\cdots B_L,
       \qquad A_l^\dagger T_l=0.

    Consequently vector algebra is just local dense-array algebra.  Applying
    an MPO uses the ordinary left/right Hamiltonian environments plus the two
    one-insertion environments of Eq. (S9), giving the advertised
    :math:`O(LD^3dw)` cost without an SVD or bond-dimension growth.
    """

    def __init__(
        self,
        reference,
        target_qntot=None,
        parallel_workers: int = 1,
        pipeline_source_workers: int = 2,
    ):
        if not isinstance(parallel_workers, int) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if (
            not isinstance(pipeline_source_workers, int)
            or pipeline_source_workers < 1
        ):
            raise ValueError("pipeline_source_workers must be a positive integer")
        self.parallel_workers = int(parallel_workers)
        self.pipeline_source_workers = int(pipeline_source_workers)
        self._executor = None
        self._operator_environment_cache = {}
        self.last_apply_timing_s = {}
        if any(tensor.ndim != 3 for tensor in reference):
            raise NotImplementedError(
                "MpsTangentSpace currently supports ordinary rank-3 MPS tensors only"
            )

        # Fix the canonical gauges once.  Keeping these arrays stable is
        # important: independently recomputing a QR gauge between Krylov steps
        # would rotate the tangent coordinates.
        right = reference.copy()
        # The round trip through both canonical directions also removes bond
        # directions that are redundant only when viewed from the opposite
        # edge.  This mirrors the preparation used by Renormalizer's TDA code.
        right.ensure_right_canonical().canonicalise()
        right.normalize("mps_and_coeff").canonicalise()
        self.reference = right
        self.right = [asnumpy(tensor.array).copy() for tensor in right]

        # Use Renormalizer's QN-resolved SVD/QR sweep rather than a dense SVD.
        # Besides preserving exact structural zeros, this supplies the
        # left-system charge attached to every A-bond.  ``right`` analogously
        # stores right-system charges on every B-bond.
        left_reference = right.copy().canonicalise()
        if not left_reference.is_left_canonical:
            raise ValueError("reference MPS could not be left canonicalized")
        self.left = [asnumpy(tensor.array).copy() for tensor in left_reference]

        self.reference_qntot = np.asarray(right.qntot, dtype=int).reshape(-1).copy()
        if target_qntot is None:
            target_qntot = self.reference_qntot
        self.target_qntot = np.asarray(target_qntot, dtype=int).reshape(-1).copy()
        if self.target_qntot.shape != self.reference_qntot.shape:
            raise ValueError(
                "target_qntot and the reference MPS quantum number have "
                "different sizes"
            )
        # ``qntot`` remains the public sector label, now referring to the
        # tangent vector rather than necessarily to the reference state.
        self.qntot = self.target_qntot
        self.qn_size = int(self.qntot.size)
        self.left_bond_qn = [
            np.asarray(qn, dtype=int).copy() for qn in left_reference.qn
        ]
        self.right_bond_qn = [np.asarray(qn, dtype=int).copy() for qn in right.qn]

        # MPS metadata stores the last bond as the charge of the empty right
        # block, even for a left-canonical state (qnidx == site_num - 1).
        # The A^dagger T gauge constraint at the last site instead needs the
        # charge of the complete LEFT block. Without this conversion a
        # nonzero-charge tangent space incorrectly retains the reference.
        self.left_bond_qn[-1] = self.reference_qntot.reshape(1, -1).copy()

        self.shapes = [
            (left.shape[0], left.shape[1], right_tensor.shape[-1])
            for left, right_tensor in zip(self.left, self.right)
        ]
        for site, (left_tensor, right_tensor) in enumerate(
            zip(self.left, self.right)
        ):
            if left_tensor.shape[1] != right_tensor.shape[1]:
                raise ValueError(
                    f"incompatible canonical physical dimensions at site {site}"
                )

        self.qn_masks: List[np.ndarray] = []
        self.gauge_qn_masks: List[np.ndarray] = []
        self.qn_active_sizes: List[int] = []
        self.gauge_constraint_sizes: List[int] = []
        for site, shape in enumerate(self.shapes):
            sigmaqn = np.asarray(left_reference._get_sigmaqn(site), dtype=int)
            row_qn = add_outer(self.left_bond_qn[site], sigmaqn)
            qnmat = add_outer(row_qn, self.right_bond_qn[site + 1])
            mask = get_qn_mask(qnmat, self.target_qntot)
            if mask.shape != shape:
                raise ValueError(
                    f"quantum-number mask at site {site} has shape {mask.shape}, "
                    f"expected {shape}"
                )

            # A^dagger T has one left-canonical and one right-canonical bond.
            # Only pairs whose subsystem charges sum to the *target* charge
            # are admissible.  For a charged tangent sector this need not be
            # the charge of the reference MPS.
            gauge_qnmat = add_outer(
                self.left_bond_qn[site + 1], self.right_bond_qn[site + 1]
            )
            gauge_mask = get_qn_mask(gauge_qnmat, self.target_qntot)
            self.qn_masks.append(mask)
            self.gauge_qn_masks.append(gauge_mask)
            self.qn_active_sizes.append(int(np.count_nonzero(mask)))
            self.gauge_constraint_sizes.append(int(np.count_nonzero(gauge_mask)))

        if not any(self.qn_active_sizes):
            raise ValueError(
                f"target total quantum number {self.target_qntot.tolist()} has no tangent coordinates"
            )
        if self.parallel_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="renormalizer-mps-tangent",
            )

    def _parallel_map(self, function, values):
        values = list(values)
        if self._executor is None or len(values) < 2:
            return [function(value) for value in values]
        return list(self._executor.map(function, values))

    def set_parallel_workers(self, parallel_workers: int) -> None:
        if not isinstance(parallel_workers, int) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if parallel_workers == self.parallel_workers and (
            parallel_workers == 1 or self._executor is not None
        ):
            return
        self.shutdown()
        self.parallel_workers = int(parallel_workers)
        if self.parallel_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="renormalizer-mps-tangent",
            )

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    @property
    def site_num(self) -> int:
        return len(self.left)

    @property
    def dimension(self) -> int:
        return int(sum(self.qn_active_sizes) - sum(self.gauge_constraint_sizes))

    @property
    def quantum_number_aware(self) -> bool:
        """Whether local coordinates enforce the selected target sector."""

        return True

    def _check_vector(self, vector: MpsTangentVector) -> None:
        if not isinstance(vector, MpsTangentVector) or vector.space is not self:
            raise ValueError("tangent vectors belong to different MPS tangent spaces")
        if len(vector.tensors) != self.site_num:
            raise ValueError("wrong number of tangent tensors")
        for site, (tensor, shape) in enumerate(zip(vector.tensors, self.shapes)):
            if tensor.shape != shape:
                raise ValueError(
                    f"tangent tensor {site} has shape {tensor.shape}, expected {shape}"
                )
            forbidden = tensor[~self.qn_masks[site]]
            if forbidden.size and np.any(forbidden != 0):
                raise ValueError(
                    f"tangent tensor {site} contains entries outside total-charge "
                    f"sector {self.target_qntot.tolist()}"
                )

    def _mask_tensors(self, tensors: Iterable[np.ndarray]) -> List[np.ndarray]:
        tensors = list(tensors)
        if len(tensors) != self.site_num:
            raise ValueError("wrong number of tangent tensors")
        output = []
        for site, (tensor, shape, mask) in enumerate(
            zip(tensors, self.shapes, self.qn_masks)
        ):
            tensor = np.asarray(tensor)
            if tensor.shape != shape:
                raise ValueError(
                    f"tangent tensor {site} has shape {tensor.shape}, expected {shape}"
                )
            masked = np.zeros(shape, dtype=tensor.dtype)
            masked[mask] = tensor[mask]
            output.append(masked)
        return output

    def project_gauge(self, tensors: Iterable[np.ndarray]) -> List[np.ndarray]:
        """Apply ``(I - A A^dagger)`` without forming the complement of ``A``."""

        tensors = self._mask_tensors(tensors)
        def project(values):
            tensor, left_iso, shape, mask, gauge_mask = values
            parent_dim = left_iso.shape[0] * left_iso.shape[1]
            a_matrix = left_iso.reshape(parent_dim, left_iso.shape[-1])
            t_matrix = tensor.reshape(parent_dim, tensor.shape[-1])
            overlap = a_matrix.conj().T @ t_matrix
            overlap[~gauge_mask] = 0
            t_matrix = t_matrix - a_matrix @ overlap
            result = t_matrix.reshape(shape)
            result[~mask] = 0
            return result

        return self._parallel_map(
            project,
            zip(
                tensors,
                self.left,
                self.shapes,
                self.qn_masks,
                self.gauge_qn_masks,
            ),
        )

    def _operator_environments(self, mpo, operators, dtype):
        key = id(mpo)
        cached = self._operator_environment_cache.get(key)
        if cached is not None and cached[0] is mpo:
            return cached[1], cached[2]

        sentinel = np.ones((1, 1, 1), dtype=dtype)

        def build_left():
            environments = [sentinel]
            for a_tensor, operator in zip(self.left, operators):
                environments.append(
                    _left_message(environments[-1], a_tensor, operator, a_tensor)
                )
            return environments

        def build_right():
            environments = [None] * (self.site_num + 1)
            environments[-1] = sentinel
            for site in range(self.site_num - 1, -1, -1):
                environments[site] = _right_message(
                    environments[site + 1],
                    self.right[site],
                    operators[site],
                    self.right[site],
                )
            return environments

        left_env, right_env = self._parallel_map(
            lambda function: function(), (build_left, build_right)
        )
        self._operator_environment_cache[key] = (mpo, left_env, right_env)
        return left_env, right_env

    def vector(
        self, tensors: Iterable[np.ndarray], project: bool = True
    ) -> MpsTangentVector:
        tensors = self._mask_tensors(tensors)
        if project:
            tensors = self.project_gauge(tensors)
        vector = MpsTangentVector(self, tensors)
        self._check_vector(vector)
        return vector

    def zeros(self, dtype=None) -> MpsTangentVector:
        if dtype is None:
            dtype = np.result_type(*[tensor.dtype for tensor in self.right])
        return self.vector([np.zeros(shape, dtype=dtype) for shape in self.shapes])

    def random(
        self, seed: Optional[int] = None, complex_values: Optional[bool] = None
    ) -> MpsTangentVector:
        rng = np.random.default_rng(seed)
        if complex_values is None:
            complex_values = any(np.iscomplexobj(tensor) for tensor in self.right)
        tensors = []
        for shape, mask in zip(self.shapes, self.qn_masks):
            tensor = np.zeros(shape)
            tensor[mask] = rng.standard_normal(np.count_nonzero(mask))
            if complex_values:
                imaginary = np.zeros(shape)
                imaginary[mask] = rng.standard_normal(np.count_nonzero(mask))
                tensor = tensor + 1j * imaginary
            tensors.append(tensor)
        return self.vector(tensors).normalize()

    def gauge_residual(self, vector: MpsTangentVector) -> float:
        self._check_vector(vector)
        residual = 0.0
        for left_iso, tensor in zip(self.left, vector.tensors):
            a_matrix = left_iso.reshape(-1, left_iso.shape[-1])
            t_matrix = tensor.reshape(-1, tensor.shape[-1])
            residual = max(
                residual,
                float(np.linalg.norm(a_matrix.conj().T @ t_matrix)),
            )
        return residual

    def sector_residual(self, vector: MpsTangentVector) -> float:
        """Return the largest norm outside the selected total-charge sector."""

        self._check_vector(vector)
        return max(
            (
                float(np.linalg.norm(tensor[~mask]))
                for tensor, mask in zip(vector.tensors, self.qn_masks)
            ),
            default=0.0,
        )

    def apply(
        self,
        mpo,
        vector: MpsTangentVector,
        energy_shift: complex = 0.0,
    ) -> MpsTangentVector:
        r"""Return :math:`P^{1\perp}(H-E)P^{1\perp}|\Phi\rangle`.

        The two insertion-environment recurrences are the algebraic form of
        Eqs. (S8)--(S10) in arXiv:2510.07279.  No full ``MPO @ MPS`` object and
        no singular-value decomposition is constructed.
        """

        self._check_vector(vector)
        if len(mpo) != self.site_num:
            raise ValueError("MPO and tangent-space lengths differ")
        operator_qntot = np.asarray(mpo.qntot, dtype=int).reshape(-1)
        if operator_qntot.shape != self.qntot.shape or np.any(operator_qntot != 0):
            raise ValueError(
                "a tangent-space endomorphism must use a charge-neutral MPO; "
                f"received operator quantum number {operator_qntot.tolist()}"
            )
        started = time.perf_counter()
        operators = [asnumpy(tensor.array) for tensor in mpo]

        dtype = np.result_type(
            *[tensor.dtype for tensor in vector.tensors],
            *[tensor.dtype for tensor in operators],
        )
        left_env, right_env = self._operator_environments(mpo, operators, dtype)
        ordinary_environment_seconds = time.perf_counter() - started

        # The source term at every site in the two affine S9 recurrences is
        # independent.  Compute all 2L source contractions first; only the
        # subsequent propagation of their prefix/suffix sums is sequential.
        def insertion_source(task):
            direction, site = task
            if direction == "left":
                return _left_message(
                    left_env[site],
                    self.left[site],
                    operators[site],
                    vector.tensors[site],
                )
            return _right_message(
                right_env[site + 1],
                self.right[site],
                operators[site],
                vector.tensors[site],
            )

        # One-insertion environments mathcal L and mathcal R (S9).  The ket
        # switches from A to B exactly where its tangent tensor is inserted.
        def build_inserted_left(source_at):
            propagation_compute_seconds = 0.0
            inserted = [None] * (self.site_num + 1)
            for site in range(self.site_num):
                local = source_at(site)
                if site == 0:
                    inserted[site + 1] = local
                else:
                    contraction_started = time.perf_counter()
                    propagated = _left_message(
                        inserted[site],
                        self.left[site],
                        operators[site],
                        self.right[site],
                    )
                    propagation_compute_seconds += (
                        time.perf_counter() - contraction_started
                    )
                    inserted[site + 1] = local + propagated
            return inserted, propagation_compute_seconds

        def build_inserted_right(source_at):
            propagation_compute_seconds = 0.0
            inserted = [None] * (self.site_num + 1)
            for site in range(self.site_num - 1, -1, -1):
                local = source_at(site)
                if site == self.site_num - 1:
                    inserted[site] = local
                else:
                    contraction_started = time.perf_counter()
                    propagated = _right_message(
                        inserted[site + 1],
                        self.right[site],
                        operators[site],
                        self.left[site],
                    )
                    propagation_compute_seconds += (
                        time.perf_counter() - contraction_started
                    )
                    inserted[site] = local + propagated
            return inserted, propagation_compute_seconds

        insertion_started = time.perf_counter()
        if self._executor is not None and self.parallel_workers >= 4:
            # Pipeline the independent source contractions over l with the two
            # ordered prefix/suffix recurrences.  The two recurrence tasks
            # occupy two workers; all remaining workers stay available for
            # future source terms.  Boundary-first submission keeps both
            # recurrences fed without changing any contraction or sum order.
            left_sources = [Future() for _ in range(self.site_num)]
            right_sources = [Future() for _ in range(self.site_num)]
            left_recurrence = self._executor.submit(
                build_inserted_left, lambda site: left_sources[site].result()
            )
            right_recurrence = self._executor.submit(
                build_inserted_right, lambda site: right_sources[site].result()
            )

            def compute_and_publish(task):
                direction, site = task
                destination = (
                    left_sources[site]
                    if direction == "left"
                    else right_sources[site]
                )
                try:
                    value = insertion_source(task)
                except BaseException as exception:
                    destination.set_exception(exception)
                    raise
                destination.set_result(value)

            source_tasks = []
            for level in range(self.site_num):
                source_tasks.append(("left", level))
                source_tasks.append(("right", self.site_num - 1 - level))
            # These dense D^3 contractions are memory-bandwidth intensive.
            # A small number of source streams balances the two propagation
            # streams; letting every spare worker run a source contraction
            # slows the critical recurrences at moderate/large worker counts.
            source_worker_count = min(
                self.pipeline_source_workers, self.parallel_workers - 2
            )
            source_queue = SimpleQueue()
            for task in source_tasks:
                source_queue.put(task)
            for _ in range(source_worker_count):
                source_queue.put(None)
            source_errors = []

            def source_runner(_runner_index):
                while True:
                    task = source_queue.get()
                    if task is None:
                        return
                    try:
                        compute_and_publish(task)
                    except BaseException as exception:
                        # Keep draining the queue so every recurrence future is
                        # resolved and an error cannot deadlock another worker.
                        source_errors.append(exception)

            source_work = [
                self._executor.submit(source_runner, runner_index)
                for runner_index in range(source_worker_count)
            ]
            for work in source_work:
                work.result()
            insertion_source_seconds = time.perf_counter() - insertion_started
            if source_errors:
                # Both recurrence tasks have now either completed or observed
                # the published exception; retrieve them before reraising.
                for recurrence in (left_recurrence, right_recurrence):
                    try:
                        recurrence.result()
                    except BaseException:
                        pass
                raise source_errors[0]
            left_result = left_recurrence.result()
            right_result = right_recurrence.result()
            inserted_left, left_propagation_seconds = left_result
            inserted_right, right_propagation_seconds = right_result
            insertion_environment_seconds = time.perf_counter() - insertion_started
            insertion_propagation_seconds = max(
                left_propagation_seconds, right_propagation_seconds
            )
            insertion_pipeline_overlap_seconds = max(
                0.0,
                insertion_source_seconds
                + insertion_propagation_seconds
                - insertion_environment_seconds,
            )
        else:
            source_tasks = [
                (direction, site)
                for direction in ("left", "right")
                for site in range(self.site_num)
            ]
            source_values = self._parallel_map(insertion_source, source_tasks)
            local_left = source_values[: self.site_num]
            local_right = source_values[self.site_num :]
            insertion_source_seconds = time.perf_counter() - insertion_started

            propagation_started = time.perf_counter()
            left_result, right_result = self._parallel_map(
                lambda item: item[0](item[1]),
                (
                    (build_inserted_left, lambda site: local_left[site]),
                    (build_inserted_right, lambda site: local_right[site]),
                ),
            )
            inserted_left, _left_compute_seconds = left_result
            inserted_right, _right_compute_seconds = right_result
            insertion_propagation_seconds = (
                time.perf_counter() - propagation_started
            )
            insertion_environment_seconds = time.perf_counter() - insertion_started
            insertion_pipeline_overlap_seconds = 0.0

        def contraction(task):
            site, order = task
            if order == 0:
                return _effective_one_site(
                    left_env[site],
                    operators[site],
                    right_env[site + 1],
                    vector.tensors[site],
                )
            if order == 1:
                return _effective_one_site(
                    inserted_left[site],
                    operators[site],
                    right_env[site + 1],
                    self.right[site],
                )
            return _effective_one_site(
                left_env[site],
                operators[site],
                inserted_right[site + 1],
                self.left[site],
            )

        local_started = time.perf_counter()
        tasks = []
        for site in range(self.site_num):
            tasks.append((site, 0))
            if site > 0:
                tasks.append((site, 1))
            if site + 1 < self.site_num:
                tasks.append((site, 2))
        values = self._parallel_map(contraction, tasks)
        grouped = [[] for _ in range(self.site_num)]
        for (site, _order), value in zip(tasks, values):
            grouped[site].append(value)
        output = []
        for site, contributions in enumerate(grouped):
            tensor = contributions[0]
            for contribution in contributions[1:]:
                tensor = tensor + contribution
            if energy_shift != 0.0:
                tensor = tensor - energy_shift * vector.tensors[site]
            output.append(tensor)
        local_contractions_seconds = time.perf_counter() - local_started

        gauge_started = time.perf_counter()
        result = self.vector(output, project=True)
        gauge_projection_seconds = time.perf_counter() - gauge_started
        self.last_apply_timing_s = {
            "ordinary_environments": ordinary_environment_seconds,
            "insertion_environments": insertion_environment_seconds,
            "insertion_source_terms": insertion_source_seconds,
            "insertion_propagation": insertion_propagation_seconds,
            "insertion_pipeline_overlap": insertion_pipeline_overlap_seconds,
            "insertion_pipeline_source_workers": (
                source_worker_count
                if self._executor is not None and self.parallel_workers >= 4
                else 0
            ),
            "local_tangent_outputs": local_contractions_seconds,
            "gauge_projection": gauge_projection_seconds,
            "total": time.perf_counter() - started,
        }
        return result

    def project_reference(self, mpo) -> MpsTangentVector:
        r"""Project ``mpo |reference>`` into this tangent sector.

        This is Eqs. (S11)--(S12) of arXiv:2510.07279 generalized from a sum
        of local operators to an MPO.  It is the charged-sector initialization
        needed for, e.g., a spin, particle-addition, or particle-removal
        operator.  The contraction leaves one tangent bra tensor open at a
        time and never constructs the bond-enlarged ``mpo @ reference`` MPS.
        """

        if len(mpo) != self.site_num:
            raise ValueError("MPO and tangent-space lengths differ")
        operator_qntot = np.asarray(mpo.qntot, dtype=int).reshape(-1)
        if operator_qntot.shape != self.reference_qntot.shape:
            raise ValueError("MPO and reference quantum-number sizes differ")
        expected = self.reference_qntot + operator_qntot
        if np.any(expected != self.target_qntot):
            raise ValueError(
                "MPO maps the reference to quantum number "
                f"{expected.tolist()}, not the tangent target "
                f"{self.target_qntot.tolist()}"
            )

        operators = [asnumpy(tensor.array) for tensor in mpo]
        dtype = np.result_type(
            *[tensor.dtype for tensor in self.right],
            *[tensor.dtype for tensor in operators],
        )
        sentinel = np.ones((1, 1, 1), dtype=dtype)

        # The ket is one consistent right-canonical representation.  The
        # open tangent bra uses A tensors to the left and B tensors to the
        # right, exactly as in Eq. (S2).
        left_mixed = [sentinel]
        for a_tensor, b_tensor, operator in zip(
            self.left, self.right, operators
        ):
            left_mixed.append(
                _left_message(left_mixed[-1], a_tensor, operator, b_tensor)
            )

        right_env = [None] * (self.site_num + 1)
        right_env[-1] = sentinel
        for site in range(self.site_num - 1, -1, -1):
            right_env[site] = _right_message(
                right_env[site + 1],
                self.right[site],
                operators[site],
                self.right[site],
            )

        cotangents = [
            _effective_one_site(
                left_mixed[site],
                operators[site],
                right_env[site + 1],
                self.right[site],
            )
            for site in range(self.site_num)
        ]
        return self.vector(cotangents, project=True)

    def to_dense(self, vector: MpsTangentVector) -> np.ndarray:
        self._check_vector(vector)
        dense = None
        for site, tangent in enumerate(vector.tensors):
            tensors = self.left[:site] + [tangent] + self.right[site + 1 :]
            block = tensors[0][0]
            for tensor in tensors[1:]:
                block = np.tensordot(block, tensor, axes=([-1], [0]))
            component = np.asarray(block)[..., 0].reshape(-1)
            dense = component if dense is None else dense + component
        if dense is None:
            return np.array([], dtype=float)
        return dense
