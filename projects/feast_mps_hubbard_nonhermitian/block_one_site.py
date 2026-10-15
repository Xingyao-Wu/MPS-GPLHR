from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from renormalizer.mps.block_env import (
    BlockEnvData,
    _arr,
    _env_diag_blocks,
    _mpo_lr_blocks,
    _mpo_lr_blocks_from_keys,
    _precompute_left_half,
    _qn_blocks,
    _thin_hop_operator,
    _contract_hop_two_site_lhalf,
)


Qn = Tuple[int, ...]


@dataclass(frozen=True)
class OneSiteCenterBlock:
    """One full QN block in the packed one-site center."""

    left_indices: np.ndarray
    right_indices: np.ndarray
    vector_indices: np.ndarray
    matrix_shape: Tuple[int, int]


class OneSiteCenterLayout:
    """Map ``(left QN, physical index, right QN)`` blocks to a packed vector."""

    def __init__(
        self,
        qn_mask: np.ndarray,
        left_qn: np.ndarray,
        right_qn: np.ndarray,
    ):
        self.qn_mask = np.asarray(qn_mask, dtype=bool)
        if self.qn_mask.ndim != 3:
            raise ValueError(
                "a one-site qn_mask must have shape (left, physical, right)"
            )
        self.shape = self.qn_mask.shape
        left_qn = np.asarray(left_qn)
        right_qn = np.asarray(right_qn)
        if len(left_qn) != self.shape[0] or len(right_qn) != self.shape[2]:
            raise ValueError("MPS bond QNs do not match the one-site center shape")

        self.size = int(np.count_nonzero(self.qn_mask))
        flat_rank = np.full(self.qn_mask.size, -1, dtype=np.intp)
        active_flat = np.flatnonzero(self.qn_mask.ravel())
        flat_rank[active_flat] = np.arange(self.size, dtype=np.intp)

        self.blocks: Dict[Tuple[Qn, int, Qn], OneSiteCenterBlock] = {}
        for left_label, left_indices in _qn_blocks(left_qn).items():
            for right_label, right_indices in _qn_blocks(right_qn).items():
                for physical in range(self.shape[1]):
                    block_mask = self.qn_mask[
                        np.ix_(left_indices, [physical], right_indices)
                    ][:, 0, :]
                    if not np.any(block_mask):
                        continue
                    if not np.all(block_mask):
                        raise NotImplementedError(
                            "partial QN blocks are not supported"
                        )
                    flat_indices = np.ravel_multi_index(
                        np.ix_(left_indices, [physical], right_indices),
                        self.shape,
                    ).reshape(-1)
                    self.blocks[(left_label, physical, right_label)] = (
                        OneSiteCenterBlock(
                            left_indices=left_indices,
                            right_indices=right_indices,
                            vector_indices=flat_rank[flat_indices],
                            matrix_shape=(len(left_indices), len(right_indices)),
                        )
                    )

        covered = sum(block.vector_indices.size for block in self.blocks.values())
        if covered != self.size:
            raise RuntimeError("one-site block layout does not cover the QN mask")

    def get(
        self,
        left_qn: Qn,
        physical: int,
        right_qn: Qn,
    ) -> Optional[OneSiteCenterBlock]:
        return self.blocks.get((left_qn, physical, right_qn))

    def pack(self, center: np.ndarray) -> np.ndarray:
        """Pack the active blocks of a dense one-site center."""
        center = _arr(center)
        if center.shape != self.shape:
            raise ValueError(
                f"center has shape {center.shape}, expected {self.shape}"
            )
        return np.asarray(center)[self.qn_mask]

    def unpack(self, center_vector: np.ndarray) -> np.ndarray:
        """Expand a packed vector, leaving the inactive QN sector zero."""
        center_vector = np.asarray(center_vector)
        if center_vector.ndim != 1 or center_vector.size != self.size:
            raise ValueError(
                f"packed center has size {center_vector.size}, expected {self.size}"
            )
        center = np.zeros(self.shape, dtype=center_vector.dtype)
        center[self.qn_mask] = center_vector
        return center


@dataclass(frozen=True)
class PackedOneSiteConnection:
    """A precontracted QN path from one packed center block to another."""

    input_indices: np.ndarray
    input_shape: Tuple[int, int]
    left_half: np.ndarray
    right_block: np.ndarray


@dataclass(frozen=True)
class PackedOneSiteOutput:
    """All contractions accumulated into one packed output block."""

    output_indices: np.ndarray
    output_shape: Tuple[int, int]
    connections: Tuple[PackedOneSiteConnection, ...]


class BlockHopOneSite:
    

    def __init__(
        self,
        left_env: BlockEnvData,
        right_env: BlockEnvData,
        local_mpo,
        center_shape,
        qn_mask,
        mps_qn_left,
        mps_qn_right,
        mpo_qn_left,
        mpo_qn_right,
        threshold: float = 1.0e-14,
        op_keys=None,
    ):
        if not isinstance(left_env, BlockEnvData) or not isinstance(
            right_env, BlockEnvData
        ):
            raise TypeError(
                "BlockHopOneSite requires block-sparse left and right environments"
            )

        self.center_shape = tuple(center_shape)
        if len(self.center_shape) != 3:
            raise ValueError(
                "a one-site center must have shape (left, physical, right)"
            )
        qn_mask = np.asarray(qn_mask, dtype=bool)
        if qn_mask.shape != self.center_shape:
            raise ValueError("qn_mask and center_shape do not match")

        local_mpo_array = _arr(local_mpo)
        expected_mpo_shape = (
            len(mpo_qn_left),
            self.center_shape[1],
            self.center_shape[1],
            len(mpo_qn_right),
        )
        if local_mpo_array.shape != expected_mpo_shape:
            raise ValueError(
                f"local MPO has shape {local_mpo_array.shape}, "
                f"expected {expected_mpo_shape}"
            )

        self.dtype = np.dtype(
            np.result_type(left_env.dtype, right_env.dtype, local_mpo_array)
        )
        self.center_layout = OneSiteCenterLayout(
            qn_mask,
            mps_qn_left,
            mps_qn_right,
        )

        mpo_left_blocks = _qn_blocks(mpo_qn_left)
        mpo_right_blocks = _qn_blocks(mpo_qn_right)
        if op_keys is None:
            operator_map = _mpo_lr_blocks(
                local_mpo_array,
                mpo_left_blocks,
                mpo_right_blocks,
                threshold,
            )
        else:
            operator_map = _mpo_lr_blocks_from_keys(
                local_mpo_array,
                mpo_left_blocks,
                mpo_right_blocks,
                op_keys,
                threshold,
            )

        centers_by_left_physical = defaultdict(list)
        for (left_label, physical, right_label), block in (
            self.center_layout.blocks.items()
        ):
            centers_by_left_physical[(left_label, physical)].append(
                (right_label, block)
            )

        right_by_operator_input = defaultdict(list)
        for (output_label, operator_label, input_label), block in (
            right_env.blocks.items()
        ):
            right_by_operator_input[(operator_label, input_label)].append(
                (output_label, block)
            )

        output_groups = {}
        for (output_left_label, mpo_left_label, input_left_label), left_block in (
            left_env.blocks.items()
        ):
            for output_physical in range(self.center_shape[1]):
                for input_physical in range(self.center_shape[1]):
                    operator_blocks = operator_map.get(
                        (mpo_left_label, output_physical, input_physical),
                        (),
                    )
                    if not operator_blocks:
                        continue
                    input_blocks = centers_by_left_physical.get(
                        (input_left_label, input_physical),
                        (),
                    )
                    if not input_blocks:
                        continue

                    for mpo_right_label, _mpo_indices, operator_block in (
                        operator_blocks
                    ):
                        for input_right_label, input_block in input_blocks:
                            right_blocks = right_by_operator_input.get(
                                (mpo_right_label, input_right_label),
                                (),
                            )
                            for output_right_label, right_block in right_blocks:
                                output_block = self.center_layout.get(
                                    output_left_label,
                                    output_physical,
                                    output_right_label,
                                )
                                if output_block is None:
                                    continue
                                thinned = _thin_hop_operator(
                                    left_block,
                                    operator_block,
                                    right_block,
                                    threshold,
                                )
                                if thinned is None:
                                    continue
                                thin_left, thin_operator, thin_right = thinned
                                left_half = _precompute_left_half(
                                    thin_left,
                                    thin_operator,
                                )
                                output_key = (
                                    output_left_label,
                                    output_physical,
                                    output_right_label,
                                )
                                group = output_groups.get(output_key)
                                if group is None:
                                    group = [output_block, []]
                                    output_groups[output_key] = group
                                group[1].append(
                                    PackedOneSiteConnection(
                                        input_indices=input_block.vector_indices,
                                        input_shape=input_block.matrix_shape,
                                        left_half=left_half,
                                        right_block=thin_right,
                                    )
                                )

        self.packed_output_groups = tuple(
            PackedOneSiteOutput(
                output_indices=block.vector_indices,
                output_shape=block.matrix_shape,
                connections=tuple(connections),
            )
            for block, connections in output_groups.values()
        )

    @property
    def size(self) -> int:
        """Dimension of the active local QN sector."""
        return self.center_layout.size

    def apply_packed(
        self,
        center_vector,
        *,
        operator_scale=1.0,
        identity_scale=0.0,
    ) -> np.ndarray:
        
        center_vector = np.asarray(center_vector)
        if center_vector.ndim != 1 or center_vector.size != self.size:
            raise ValueError(
                f"packed center has size {center_vector.size}, expected {self.size}"
            )
        if np.ndim(operator_scale) != 0 or np.ndim(identity_scale) != 0:
            raise ValueError("operator_scale and identity_scale must be scalars")

        dtype = np.result_type(
            self.dtype,
            center_vector.dtype,
            np.asarray(operator_scale).dtype,
            np.asarray(identity_scale).dtype,
        )
        if identity_scale == 0:
            output = np.zeros(self.size, dtype=dtype)
        else:
            output = np.asarray(identity_scale * center_vector, dtype=dtype)

        skip_operator = operator_scale == 0
        unit_operator = operator_scale == 1
        if skip_operator:
            return output

        for output_group in self.packed_output_groups:
            output_block = np.zeros(output_group.output_shape, dtype=dtype)
            for connection in output_group.connections:
                input_block = center_vector[connection.input_indices].reshape(
                    connection.input_shape
                )
                output_block += _contract_hop_two_site_lhalf(
                    connection.left_half,
                    connection.right_block,
                    input_block,
                ).reshape(output_group.output_shape)
            if not unit_operator:
                output_block *= operator_scale
            output[output_group.output_indices] += output_block.reshape(-1)
        return output

    def apply_shifted_packed(self, center_vector, shift) -> np.ndarray:
        """Apply the FEAST local operator ``shift * I - H_eff``."""
        return self.apply_packed(
            center_vector,
            operator_scale=-1.0,
            identity_scale=shift,
        )

    def __call__(self, center) -> np.ndarray:
        """Dense-shape debugging interface, with inactive QN entries zero."""
        packed = self.center_layout.pack(center)
        return self.center_layout.unpack(self.apply_packed(packed))


def block_hop_expr_one_site(*args, **kwargs) -> BlockHopOneSite:
    """Construct a compiled packed one-site effective-MPO action."""
    return BlockHopOneSite(*args, **kwargs)


def block_hdiag_one_site(
    left_env: BlockEnvData,
    right_env: BlockEnvData,
    local_mpo,
    center_shape,
    mpo_qn_left,
    mpo_qn_right,
    threshold: float = 1.0e-14,
    op_keys=None,
) -> np.ndarray:
    """Return the dense-shape diagonal of a one-site effective MPO.

    The result equals ``L[a,b,a] W[b,d,d,f] R[l,f,l]``.  Callers that keep
    local vectors packed should select ``result[qn_mask]`` once during local
    solver setup.
    """
    if not isinstance(left_env, BlockEnvData) or not isinstance(
        right_env, BlockEnvData
    ):
        raise TypeError(
            "block_hdiag_one_site requires block-sparse left and right environments"
        )
    center_shape = tuple(center_shape)
    if len(center_shape) != 3:
        raise ValueError(
            "a one-site center must have shape (left, physical, right)"
        )
    local_mpo_array = _arr(local_mpo)
    expected_mpo_shape = (
        len(mpo_qn_left),
        center_shape[1],
        center_shape[1],
        len(mpo_qn_right),
    )
    if local_mpo_array.shape != expected_mpo_shape:
        raise ValueError(
            f"local MPO has shape {local_mpo_array.shape}, "
            f"expected {expected_mpo_shape}"
        )

    mpo_left_blocks = _qn_blocks(mpo_qn_left)
    mpo_right_blocks = _qn_blocks(mpo_qn_right)
    if op_keys is None:
        operator_map = _mpo_lr_blocks(
            local_mpo_array,
            mpo_left_blocks,
            mpo_right_blocks,
            threshold,
        )
    else:
        operator_map = _mpo_lr_blocks_from_keys(
            local_mpo_array,
            mpo_left_blocks,
            mpo_right_blocks,
            op_keys,
            threshold,
        )

    dtype = np.result_type(left_env.dtype, right_env.dtype, local_mpo_array)
    diagonal = np.zeros(center_shape, dtype=dtype)
    left_diagonal_by_mpo = _env_diag_blocks(left_env)
    right_diagonal_by_mpo = _env_diag_blocks(right_env)

    for mpo_left_label, left_items in left_diagonal_by_mpo.items():
        for physical in range(center_shape[1]):
            operator_blocks = operator_map.get(
                (mpo_left_label, physical, physical),
                (),
            )
            for mpo_right_label, _indices, operator_block in operator_blocks:
                right_items = right_diagonal_by_mpo.get(mpo_right_label, ())
                if not right_items:
                    continue
                for _left_label, left_indices, left_diagonal in left_items:
                    left_operator = left_diagonal @ operator_block
                    for _right_label, right_indices, right_diagonal in right_items:
                        diagonal[
                            np.ix_(left_indices, [physical], right_indices)
                        ] += (left_operator @ right_diagonal.T)[:, None, :]
    return diagonal


__all__ = [
    "BlockHopOneSite",
    "OneSiteCenterBlock",
    "OneSiteCenterLayout",
    "block_hdiag_one_site",
    "block_hop_expr_one_site",
]
