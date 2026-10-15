"""Small exact SU(2) representation layer for reduced tangent tensors.

Renormalizer's historical quantum numbers are additive Abelian labels.  This
module supplies the extra information that a non-Abelian tensor needs:
irreducible representations, outer multiplicities, fusion channels, and
Clebsch--Gordan reconstruction.  It intentionally does not pretend that a
``two_j`` label is an Abelian charge.

Twice-spin integers are used throughout, so half-integer irreps never depend
on floating-point comparisons.  Dense basis states may carry phases; this is
needed when an MPS canonicalization chooses a different Condon--Shortley gauge
on a virtual multiplet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan


def _validate_two_j(two_j: int) -> int:
    two_j = int(two_j)
    if two_j < 0:
        raise ValueError("twice-spin labels must be non-negative")
    return two_j


def _validate_two_m(two_j: int, two_m: int) -> int:
    two_m = int(two_m)
    if abs(two_m) > two_j or (two_j - two_m) % 2:
        raise ValueError(f"2m={two_m} is not in the j={two_j}/2 irrep")
    return two_m


@dataclass(frozen=True)
class SU2Multiplet:
    """An SU(2) irrep and its outer (degeneracy-space) multiplicity."""

    two_j: int
    degeneracy: int = 1

    def __post_init__(self):
        object.__setattr__(self, "two_j", _validate_two_j(self.two_j))
        if int(self.degeneracy) <= 0:
            raise ValueError("multiplet degeneracy must be positive")
        object.__setattr__(self, "degeneracy", int(self.degeneracy))


@dataclass(frozen=True)
class SU2State:
    """One magnetic state in a particular outer-multiplicity copy."""

    two_j: int
    alpha: int
    two_m: int
    phase: complex = 1.0


class SU2Basis:
    r"""Ordered dense realization of ``direct_sum_j D_j tensor V_j``.

    Parameters
    ----------
    multiplets
        Sequence of :class:`SU2Multiplet`, integer ``two_j`` labels, or
        ``(two_j, degeneracy)`` pairs.  Magnetic states are ordered from
        ``m=+j`` to ``m=-j`` inside each multiplet copy.
    phases
        Optional unit-modulus phase for each resulting dense basis state.
        If ``|i>_dense = phase[i] |j,alpha,m>_CS``, the reconstruction uses
        the inverse phase.  This makes virtual-bond gauge conventions
        explicit rather than silently treating them as new quantum numbers.
    state_order
        Optional explicit dense ordering of ``(two_j, alpha, two_m)`` states.
        It must be a permutation of the states declared by ``multiplets``.
    """

    def __init__(
        self,
        multiplets: Iterable,
        phases: Sequence[complex] = None,
        state_order: Sequence[Tuple[int, int, int]] = None,
    ):
        parsed: List[SU2Multiplet] = []
        for item in multiplets:
            if isinstance(item, SU2Multiplet):
                parsed.append(item)
            elif np.isscalar(item):
                parsed.append(SU2Multiplet(int(item)))
            else:
                parsed.append(SU2Multiplet(*item))
        if not parsed:
            raise ValueError("an SU2Basis needs at least one multiplet")

        raw_states = []
        alpha_count = {}
        copies = []
        for multiplet in parsed:
            start = alpha_count.get(multiplet.two_j, 0)
            for offset in range(multiplet.degeneracy):
                alpha = start + offset
                copies.append((multiplet.two_j, alpha))
                for two_m in range(multiplet.two_j, -multiplet.two_j - 1, -2):
                    raw_states.append((multiplet.two_j, alpha, two_m))
            alpha_count[multiplet.two_j] = start + multiplet.degeneracy

        if state_order is not None:
            state_order = [tuple(int(value) for value in state) for state in state_order]
            if len(state_order) != len(raw_states) or set(state_order) != set(raw_states):
                raise ValueError(
                    "state_order must contain every (two_j, alpha, two_m) "
                    "state exactly once"
                )
            raw_states = state_order

        if phases is None:
            phases = np.ones(len(raw_states), dtype=complex)
        phases = np.asarray(phases, dtype=complex).reshape(-1)
        if len(phases) != len(raw_states):
            raise ValueError(
                f"received {len(phases)} phases for an SU(2) basis of "
                f"dimension {len(raw_states)}"
            )
        if np.any(np.abs(np.abs(phases) - 1.0) > 1.0e-12):
            raise ValueError("SU(2) basis phases must have unit modulus")

        self.multiplets = tuple(parsed)
        self.copies = tuple(copies)
        self.states = tuple(
            SU2State(two_j, alpha, two_m, complex(phase))
            for (two_j, alpha, two_m), phase in zip(raw_states, phases)
        )

    @classmethod
    def spin(
        cls,
        two_j: int,
        phases: Sequence[complex] = None,
        state_order: Sequence[Tuple[int, int, int]] = None,
    ) -> "SU2Basis":
        return cls(
            [SU2Multiplet(two_j)], phases=phases, state_order=state_order
        )

    @classmethod
    def trivial(cls, degeneracy: int = 1) -> "SU2Basis":
        return cls([SU2Multiplet(0, degeneracy)])

    @property
    def dimension(self) -> int:
        return len(self.states)

    def degeneracy(self, two_j: int) -> int:
        """Return the outer multiplicity of an irrep on this axis."""

        two_j = int(two_j)
        return sum(item.degeneracy for item in self.multiplets if item.two_j == two_j)

    def state_index(self, two_j: int, alpha: int, two_m: int) -> int:
        key = (int(two_j), int(alpha), int(two_m))
        for index, state in enumerate(self.states):
            if (state.two_j, state.alpha, state.two_m) == key:
                return index
        raise KeyError(key)

    def dual(self) -> "SU2Basis":
        r"""Return the contragredient axis in Condon--Shortley convention.

        SU(2) irreps are self-dual, but the magnetic labels and phases of an
        incoming tensor leg are not identical to those of an outgoing leg:
        ``|j,m>_dual = (-1)^(j-m) |j,-m>``.  Keeping this conversion explicit
        prevents the common error of treating leg orientation as irrelevant.
        """

        order = []
        phases = []
        for state in self.states:
            order.append((state.two_j, state.alpha, -state.two_m))
            exponent = (state.two_j + state.two_m) // 2
            phases.append(state.phase * ((-1) ** exponent))
        return type(self)(self.multiplets, phases=phases, state_order=order)

    def __repr__(self):
        content = ", ".join(
            f"j={item.two_j}/2 x {item.degeneracy}" for item in self.multiplets
        )
        return f"SU2Basis({content})"


@dataclass(frozen=True)
class SU2FusionChannel:
    """One degeneracy/fusion-tree coordinate of a reduced tensor."""

    axis_two_j: Tuple[int, ...]
    axis_alpha: Tuple[int, ...]
    intermediate_two_j: Tuple[int, ...]
    target_two_j: int


def allowed_fusions(two_j1: int, two_j2: int) -> Tuple[int, ...]:
    """Return the exact SU(2) fusion rule ``j1 tensor j2``."""

    two_j1 = _validate_two_j(two_j1)
    two_j2 = _validate_two_j(two_j2)
    return tuple(range(abs(two_j1 - two_j2), two_j1 + two_j2 + 1, 2))


@lru_cache(maxsize=None)
def _cg(two_j1, two_m1, two_j2, two_m2, two_j3, two_m3) -> float:
    if two_m1 + two_m2 != two_m3:
        return 0.0
    value = clebsch_gordan(
        Rational(two_j1, 2),
        Rational(two_j2, 2),
        Rational(two_j3, 2),
        Rational(two_m1, 2),
        Rational(two_m2, 2),
        Rational(two_m3, 2),
    )
    return float(value)


def _intermediate_paths(axis_two_j, target_two_j):
    if len(axis_two_j) == 1:
        if axis_two_j[0] == target_two_j:
            yield ()
        return

    def recurse(position, current, path):
        if position == len(axis_two_j):
            if current == target_two_j:
                yield tuple(path)
            return
        for fused in allowed_fusions(current, axis_two_j[position]):
            if position == len(axis_two_j) - 1 and fused != target_two_j:
                continue
            yield from recurse(position + 1, fused, path + [fused])

    yield from recurse(1, axis_two_j[0], [])


def su2_coupled_embedding(
    axes: Sequence[SU2Basis], target_two_j: int, target_two_m: int
):
    r"""Return the CG embedding from reduced channels to a dense tensor.

    Columns correspond to outer-degeneracy choices and a left-associated
    fusion tree.  They are orthonormal for a fixed ``(J,M)`` component.  The
    same ordered channel list is obtained for every magnetic component, which
    is the concrete Wigner--Eckart reduction used by the tangent wrappers.
    """

    axes = tuple(axes)
    if not axes:
        raise ValueError("at least one SU(2) axis is required")
    if not all(isinstance(axis, SU2Basis) for axis in axes):
        raise TypeError("all axes must be SU2Basis instances")
    target_two_j = _validate_two_j(target_two_j)
    target_two_m = _validate_two_m(target_two_j, target_two_m)

    channels: List[SU2FusionChannel] = []
    for copies in product(*(axis.copies for axis in axes)):
        axis_two_j = tuple(copy[0] for copy in copies)
        axis_alpha = tuple(copy[1] for copy in copies)
        for path in _intermediate_paths(axis_two_j, target_two_j):
            channels.append(
                SU2FusionChannel(
                    axis_two_j,
                    axis_alpha,
                    tuple(path),
                    target_two_j,
                )
            )

    shape = tuple(axis.dimension for axis in axes)
    embedding = np.zeros((int(np.prod(shape)), len(channels)), dtype=complex)
    for column, channel in enumerate(channels):
        state_lists = []
        for axis, two_j, alpha in zip(
            axes, channel.axis_two_j, channel.axis_alpha
        ):
            state_lists.append(
                [
                    (index, state)
                    for index, state in enumerate(axis.states)
                    if state.two_j == two_j and state.alpha == alpha
                ]
            )

        for selected in product(*state_lists):
            indices = tuple(item[0] for item in selected)
            states = tuple(item[1] for item in selected)
            if sum(state.two_m for state in states) != target_two_m:
                continue
            coefficient = 1.0
            current_j = states[0].two_j
            current_m = states[0].two_m
            for position in range(1, len(states)):
                next_j = channel.intermediate_two_j[position - 1]
                next_m = current_m + states[position].two_m
                coefficient *= _cg(
                    current_j,
                    current_m,
                    states[position].two_j,
                    states[position].two_m,
                    next_j,
                    next_m,
                )
                current_j, current_m = next_j, next_m
            # Dense states may differ from the Condon--Shortley states by a
            # phase.  Components therefore transform with the inverse phase.
            coefficient *= np.prod([state.phase.conjugate() for state in states])
            embedding[np.ravel_multi_index(indices, shape), column] = coefficient

    return embedding, tuple(channels)


@dataclass
class SU2ReducedTensor:
    """A tensor stored only by outer-degeneracy and fusion-channel data."""

    axes: Tuple[SU2Basis, ...]
    target_two_j: int
    target_two_m: int
    coefficients: np.ndarray

    def __init__(
        self,
        axes: Sequence[SU2Basis],
        target_two_j: int,
        target_two_m: int,
        coefficients,
    ):
        self.axes = tuple(axes)
        self.target_two_j = _validate_two_j(target_two_j)
        self.target_two_m = _validate_two_m(self.target_two_j, target_two_m)
        embedding, self.channels = su2_coupled_embedding(
            self.axes, self.target_two_j, self.target_two_m
        )
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (embedding.shape[1],):
            raise ValueError(
                f"reduced coefficient shape {coefficients.shape} does not match "
                f"the {embedding.shape[1]} SU(2) fusion channels"
            )
        self.coefficients = coefficients.copy()

    @property
    def shape(self):
        return tuple(axis.dimension for axis in self.axes)

    @property
    def reduced_size(self) -> int:
        return len(self.coefficients)

    @property
    def reduced_blocks(self):
        """Return conventional dense degeneracy blocks keyed by fusion data.

        A key is ``(axis_two_j, intermediate_two_j)``.  Magnetic indices are
        absent; each block axis runs only over the corresponding outer
        multiplicity ``alpha``.  This is the same reduced/structural split
        used by non-Abelian tensor libraries.
        """

        blocks = {}
        for channel, coefficient in zip(self.channels, self.coefficients):
            key = (channel.axis_two_j, channel.intermediate_two_j)
            if key not in blocks:
                shape = tuple(
                    axis.degeneracy(two_j)
                    for axis, two_j in zip(self.axes, channel.axis_two_j)
                )
                blocks[key] = np.zeros(shape, dtype=self.coefficients.dtype)
            blocks[key][channel.axis_alpha] = coefficient
        return blocks

    def to_dense(self) -> np.ndarray:
        embedding, _ = su2_coupled_embedding(
            self.axes, self.target_two_j, self.target_two_m
        )
        return (embedding @ self.coefficients).reshape(self.shape)

    def magnetic_component(self, target_two_m: int) -> "SU2ReducedTensor":
        """Reconstruct another M component with the same reduced data."""

        return type(self)(
            self.axes,
            self.target_two_j,
            target_two_m,
            self.coefficients,
        )

    @classmethod
    def from_dense(
        cls,
        axes: Sequence[SU2Basis],
        target_two_j: int,
        target_two_m: int,
        tensor,
        tolerance: float = 1.0e-10,
    ) -> "SU2ReducedTensor":
        embedding, _ = su2_coupled_embedding(
            axes, target_two_j, target_two_m
        )
        dense = np.asarray(tensor).reshape(-1)
        if dense.shape != (embedding.shape[0],):
            raise ValueError("dense tensor dimensions do not match the SU(2) axes")
        coefficients = embedding.conj().T @ dense
        residual = np.linalg.norm(dense - embedding @ coefficients)
        scale = max(1.0, float(np.linalg.norm(dense)))
        if residual > tolerance * scale:
            raise ValueError(
                f"tensor has relative weight {residual / scale:.3e} outside "
                f"the requested SU(2) (J,M)=({target_two_j}/2,{target_two_m}/2) sector"
            )
        return cls(axes, target_two_j, target_two_m, coefficients)

    @classmethod
    def from_reduced_blocks(
        cls,
        axes: Sequence[SU2Basis],
        target_two_j: int,
        target_two_m: int,
        blocks,
    ) -> "SU2ReducedTensor":
        """Construct from degeneracy blocks without introducing magnetic data."""

        _embedding, channels = su2_coupled_embedding(
            axes, target_two_j, target_two_m
        )
        coefficients = []
        expected_keys = set()
        for channel in channels:
            key = (channel.axis_two_j, channel.intermediate_two_j)
            expected_keys.add(key)
            if key not in blocks:
                raise ValueError(f"missing SU(2) reduced block {key}")
            coefficients.append(np.asarray(blocks[key])[channel.axis_alpha])
        unexpected = set(blocks) - expected_keys
        if unexpected:
            raise ValueError(f"unexpected SU(2) reduced blocks: {unexpected}")
        return cls(axes, target_two_j, target_two_m, coefficients)
