"""Non-Abelian SU(2)-reduced one-site MPS tangent vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from renormalizer.mps.tangent import MpsTangentSpace, MpsTangentVector
from renormalizer.utils.su2 import SU2Basis, su2_coupled_embedding


@dataclass
class SU2MpsTangentVector:
    r"""One fixed-M component stored in Wigner--Eckart reduced coordinates.

    The number and ordering of coordinates do not depend on M.  A scalar MPO
    therefore acts on the same reduced vector in every magnetic component.
    """

    space: "SU2MpsTangentSpace"
    coefficients: List[np.ndarray]

    def copy(self):
        return type(self)(self.space, [array.copy() for array in self.coefficients])

    def dot(self, other) -> complex:
        self.space._check_vector(other)
        return sum(
            (
                np.vdot(left, right)
                for left, right in zip(self.coefficients, other.coefficients)
            ),
            0.0j,
        )

    @property
    def norm(self) -> float:
        value = self.dot(self).real
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        return float(np.sqrt(value))

    def scale(self, value, inplace: bool = False):
        result = self if inplace else self.copy()
        result.coefficients = [value * array for array in result.coefficients]
        return result

    def axpy(self, value, other):
        self.space._check_vector(other)
        self.coefficients = [
            array + value * increment
            for array, increment in zip(self.coefficients, other.coefficients)
        ]
        return self

    def normalize(self):
        norm = self.norm
        if norm == 0.0:
            raise ValueError("cannot normalize a zero SU(2) tangent vector")
        return self.scale(1.0 / norm, inplace=True)

    def to_dense(self) -> np.ndarray:
        return self.space.to_dense(self)

    def as_abelian(self) -> MpsTangentVector:
        """Return the gauge-fixed dense-block tangent representation."""

        return self.space.as_abelian(self)


class SU2MpsTangentSpace:
    r"""SU(2)-reduced subspace of :class:`MpsTangentSpace`.

    ``site_axes[l]`` describes the dense bases of the left branch, physical
    leg, and right branch of tangent tensor ``T[l]``.  Every axis is an
    :class:`~renormalizer.utils.su2.SU2Basis` containing irrep and outer-
    multiplicity labels.  Clebsch--Gordan fusion first selects total ``(J,M)``;
    the paper's ``A^dagger T=0`` projector is then imposed.  Initialization
    rejects gauges for which these two projectors do not commute, rather than
    silently producing a tensor that is not actually SU(2) reduced.

    The wrapped Abelian tangent space may simultaneously select particle
    number or other commuting charges through ``target_qntot``.  This realizes
    product symmetries such as U(1) x SU(2).
    """

    def __init__(
        self,
        reference,
        site_axes: Sequence[Sequence[SU2Basis]],
        target_two_j: int,
        target_two_m: Optional[int] = None,
        target_qntot=None,
        tolerance: float = 1.0e-10,
    ):
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.tolerance = float(tolerance)
        self.base = MpsTangentSpace(reference, target_qntot=target_qntot)
        self.reference = self.base.reference
        self.target_qntot = self.base.target_qntot
        self.qntot = self.target_qntot
        self.target_two_j = int(target_two_j)
        if target_two_m is None:
            target_two_m = target_two_j
        self.target_two_m = int(target_two_m)
        self.site_axes = tuple(tuple(axes) for axes in site_axes)
        if len(self.site_axes) != self.base.site_num:
            raise ValueError("site_axes must contain one entry for each MPS site")

        self.embeddings: List[np.ndarray] = []
        self.fusion_channels: List[Tuple] = []
        self.coordinate_embeddings: List[np.ndarray] = []
        for site, (axes, shape) in enumerate(zip(self.site_axes, self.base.shapes)):
            if len(axes) != 3 or not all(isinstance(axis, SU2Basis) for axis in axes):
                raise TypeError(
                    "each MPS site needs three SU2Basis axes: left, physical, right"
                )
            axis_shape = tuple(axis.dimension for axis in axes)
            if axis_shape != shape:
                raise ValueError(
                    f"SU(2) axes at site {site} have shape {axis_shape}, "
                    f"but its tangent tensor has shape {shape}"
                )
            embedding, channels = su2_coupled_embedding(
                axes, self.target_two_j, self.target_two_m
            )
            self.fusion_channels.append(channels)
            self.embeddings.append(embedding)
            self.coordinate_embeddings.append(
                self._build_coordinate_embedding(site, embedding)
            )
        if not any(matrix.shape[1] for matrix in self.coordinate_embeddings):
            raise ValueError("the selected SU(2) tangent sector is empty")

    def _build_coordinate_embedding(self, site, embedding):
        mask = self.base.qn_masks[site].reshape(-1)
        selected = embedding.copy()
        selected[~mask, :] = 0
        removed = np.linalg.norm(embedding - selected)
        scale = max(1.0, float(np.linalg.norm(embedding)))
        if removed > self.tolerance * scale:
            raise ValueError(
                f"SU(2) sector at site {site} is incompatible with the selected "
                "commuting Abelian charge"
            )
        if selected.shape[1] == 0:
            return np.zeros((selected.shape[0], 0), dtype=complex)
        gram_error = np.linalg.norm(
            selected.conj().T @ selected - np.eye(selected.shape[1])
        )
        if gram_error > self.tolerance * max(1, selected.shape[1]):
            raise ValueError("SU(2) fusion columns are not orthonormal")

        shape = self.base.shapes[site]
        left_iso = self.base.left[site].reshape(-1, shape[-1])
        gauge_mask = self.base.gauge_qn_masks[site]
        candidate = np.empty_like(selected, dtype=np.result_type(selected, left_iso))
        for column in range(selected.shape[1]):
            matrix = selected[:, column].reshape(-1, shape[-1])
            overlap = left_iso.conj().T @ matrix
            overlap[~gauge_mask] = 0
            projected = matrix - left_iso @ overlap
            dense = projected.reshape(-1)
            dense[~mask] = 0
            candidate[:, column] = dense

        # For an SU(2)-adapted canonical gauge, the discarded-space projector
        # is an intertwiner and cannot leave the CG-coupled subspace.
        outside = candidate - selected @ (selected.conj().T @ candidate)
        outside_norm = np.linalg.norm(outside)
        candidate_scale = max(1.0, float(np.linalg.norm(candidate)))
        if outside_norm > self.tolerance * candidate_scale:
            raise ValueError(
                f"MPS canonical gauge at site {site} is not SU(2)-adapted; "
                f"relative intertwiner residual {outside_norm / candidate_scale:.3e}"
            )

        gram = candidate.conj().T @ candidate
        gram = 0.5 * (gram + gram.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        if not eigenvalues.size:
            return np.zeros((selected.shape[0], 0), dtype=candidate.dtype)
        cutoff = self.tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
        retained = eigenvalues > cutoff
        if not np.any(retained):
            return np.zeros((selected.shape[0], 0), dtype=candidate.dtype)
        # Work in the fusion-channel Gram matrix rather than accepting the
        # arbitrary left-vector phases of an SVD.  The resulting reduced basis
        # is tied to degeneracy/fusion labels and is shared by all M components.
        return candidate @ (
            eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
        )

    @property
    def site_num(self) -> int:
        return self.base.site_num

    @property
    def dimension(self) -> int:
        return int(sum(matrix.shape[1] for matrix in self.coordinate_embeddings))

    @property
    def non_abelian(self) -> bool:
        return True

    @property
    def reduced(self) -> bool:
        return True

    def _check_vector(self, vector):
        if not isinstance(vector, SU2MpsTangentVector) or vector.space is not self:
            raise ValueError("vectors belong to different SU(2) MPS tangent spaces")
        if len(vector.coefficients) != self.site_num:
            raise ValueError("wrong number of reduced tangent tensors")
        for site, (coefficients, embedding) in enumerate(
            zip(vector.coefficients, self.coordinate_embeddings)
        ):
            if np.asarray(coefficients).shape != (embedding.shape[1],):
                raise ValueError(f"wrong reduced-coordinate shape at site {site}")

    def vector(self, coefficients: Iterable[np.ndarray]):
        coefficients = [np.asarray(array).copy() for array in coefficients]
        vector = SU2MpsTangentVector(self, coefficients)
        self._check_vector(vector)
        return vector

    def zeros(self, dtype=complex):
        return self.vector(
            [np.zeros(matrix.shape[1], dtype=dtype) for matrix in self.coordinate_embeddings]
        )

    def random(self, seed=None, complex_values=True):
        rng = np.random.default_rng(seed)
        coefficients = []
        for matrix in self.coordinate_embeddings:
            array = rng.standard_normal(matrix.shape[1])
            if complex_values:
                array = array + 1j * rng.standard_normal(matrix.shape[1])
            coefficients.append(array)
        return self.vector(coefficients).normalize()

    def as_abelian(self, vector) -> MpsTangentVector:
        self._check_vector(vector)
        tensors = [
            (embedding @ coefficients).reshape(shape)
            for embedding, coefficients, shape in zip(
                self.coordinate_embeddings,
                vector.coefficients,
                self.base.shapes,
            )
        ]
        return self.base.vector(tensors, project=False)

    def _from_abelian(self, vector, check=True):
        self.base._check_vector(vector)
        coefficients = []
        residual_squared = 0.0
        norm_squared = 0.0
        for tensor, embedding in zip(vector.tensors, self.coordinate_embeddings):
            dense = tensor.reshape(-1)
            reduced = embedding.conj().T @ dense
            residual_squared += float(
                np.linalg.norm(dense - embedding @ reduced) ** 2
            )
            norm_squared += float(np.linalg.norm(dense) ** 2)
            coefficients.append(reduced)
        residual = np.sqrt(residual_squared)
        scale = max(1.0, np.sqrt(norm_squared))
        if check and residual > self.tolerance * scale:
            raise ValueError(
                "MPO action leaves the selected SU(2) irrep; the operator is "
                f"not an SU(2) scalar (relative residual {residual / scale:.3e})"
            )
        return self.vector(coefficients)

    def apply(self, mpo, vector, energy_shift=0.0):
        dense_result = self.base.apply(
            mpo, self.as_abelian(vector), energy_shift=energy_shift
        )
        return self._from_abelian(dense_result, check=True)

    def project_reference(self, mpo):
        return self._from_abelian(self.base.project_reference(mpo), check=True)

    def to_dense(self, vector):
        return self.as_abelian(vector).to_dense()

    def gauge_residual(self, vector):
        return self.base.gauge_residual(self.as_abelian(vector))

    def su2_residual(self, vector):
        self._check_vector(vector)
        residual = 0.0
        for coefficients, coordinate, coupled in zip(
            vector.coefficients, self.coordinate_embeddings, self.embeddings
        ):
            tensor = coordinate @ coefficients
            residual = max(
                residual,
                float(np.linalg.norm(tensor - coupled @ (coupled.conj().T @ tensor))),
            )
        return residual
