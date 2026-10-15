"""Exact GPU contractions with bounded intermediate tensors.

Slice an output MPS index and retain the full MPO. The workspace target
limits temporaries down to one output slice, which may exceed the target;
resident inputs, outputs and library workspace are separate.
Small contractions keep their existing optimized paths.
"""
import os
import numpy as np
from renormalizer.mps.backend import xp, USE_GPU


WORKSPACE_BYTES = int(os.environ.get("RENO_CONTRACTION_WORKSPACE_MB", "512")) * 2**20
# Preserve optimized constant contractions when their tensors fit comfortably.
# Slicing every moderate contraction would rebuild these constants at each
# Davidson/TD exponential action and needlessly repeat work.
UNSLICED_LIMIT_BYTES = int(os.environ.get("RENO_UNSLICED_LIMIT_MB", "4096")) * 2**20
if WORKSPACE_BYTES <= 0 or UNSLICED_LIMIT_BYTES <= 0:
    raise ValueError("contraction memory settings must be positive")


def _chunk(extent, elements_per_slice, dtype, workspace):
    return max(1, min(extent, workspace // max(1, elements_per_slice * np.dtype(dtype).itemsize)))


def needs_environment(env, bra, op, ket, domain):
    if not USE_GPU:
        return False
    dtype = xp.result_type(env, bra, op, ket)
    batch = max(env.shape[0], ket.shape[0])
    if domain == "L":
        size = batch * ket.shape[1] * bra.shape[2] * max(op.shape[0], op.shape[-1]) * max(op.shape[1:3])
    else:
        size = batch * ket.shape[-1] * bra.shape[0] * max(op.shape[0], op.shape[-1]) * max(op.shape[1:3])
    return size * np.dtype(dtype).itemsize > UNSLICED_LIMIT_BYTES


def left(env, bra, op, ket, workspace=None):
    """env[k,a,w,c], bra[a,p,b], op[w,p,q,v], ket[k,c,q,d]."""
    workspace = WORKSPACE_BYTES if workspace is None else workspace
    env, bra, op, ket = (xp.asarray(t) for t in (env, bra, op, ket))
    dtype = xp.result_type(env, bra, op, ket)
    batch = max(env.shape[0], ket.shape[0])
    out = xp.empty((batch, bra.shape[-1], op.shape[-1], ket.shape[-1]), dtype=dtype)
    per = batch * ket.shape[1] * max(op.shape[1:3]) * max(op.shape[0], op.shape[-1])
    step = _chunk(bra.shape[-1], per, dtype, workspace)
    for start in range(0, bra.shape[-1], step):
        sl = slice(start, start + step)
        t = xp.einsum("kawc,apb->kwcpb", env, bra[:, :, sl].conj(), optimize=True)
        t = xp.einsum("kwcpb,wpqv->kcbqv", t, op, optimize=True)
        out[:, sl] = xp.einsum("kcbqv,kcqd->kbvd", t, ket, optimize=True)
    return out


def right(env, bra, op, ket, workspace=None):
    """Right-to-left counterpart of left, with the same tensor conventions."""
    workspace = WORKSPACE_BYTES if workspace is None else workspace
    env, bra, op, ket = (xp.asarray(t) for t in (env, bra, op, ket))
    dtype = xp.result_type(env, bra, op, ket)
    batch = max(env.shape[0], ket.shape[0])
    out = xp.empty((batch, bra.shape[0], op.shape[0], ket.shape[1]), dtype=dtype)
    per = batch * ket.shape[-1] * max(op.shape[1:3]) * max(op.shape[0], op.shape[-1])
    step = _chunk(bra.shape[0], per, dtype, workspace)
    for start in range(0, bra.shape[0], step):
        sl = slice(start, start + step)
        t = xp.einsum("apb,kbvd->kapvd", bra[sl].conj(), env, optimize=True)
        t = xp.einsum("kapvd,wpqv->kawqd", t, op, optimize=True)
        out[:, sl] = xp.einsum("kawqd,kcqd->kawc", t, ket, optimize=True)
    return out


def effective(left_env, op, right_env, ket, workspace=None):
    workspace = WORKSPACE_BYTES if workspace is None else workspace
    left_env, op, right_env, ket = (xp.asarray(t) for t in (left_env, op, right_env, ket))
    dtype = xp.result_type(left_env, op, right_env, ket)
    batch = max(left_env.shape[0], right_env.shape[0], ket.shape[0])
    out = xp.empty((batch, left_env.shape[1], op.shape[1], right_env.shape[1]), dtype=dtype)
    per = batch * ket.shape[-1] * max(op.shape[1:3]) * max(op.shape[0], op.shape[-1])
    step = _chunk(left_env.shape[1], per, dtype, workspace)
    for start in range(0, left_env.shape[1], step):
        sl = slice(start, start + step)
        t = xp.einsum("kawc,kcqd->kawqd", left_env[:, sl], ket, optimize=True)
        t = xp.einsum("kawqd,wpqv->kapvd", t, op, optimize=True)
        out[:, sl] = xp.einsum("kapvd,kbvd->kapb", t, right_env, optimize=True)
    return out


def needs_effective(left_env, op, right_env, ket):
    if not USE_GPU:
        return False
    dtype = xp.result_type(left_env, op, right_env, ket)
    batch = max(left_env.shape[0], right_env.shape[0], ket.shape[0])
    size = batch * left_env.shape[1] * ket.shape[-1] * max(op.shape[1:3]) * max(op.shape[0], op.shape[-1])
    return size * np.dtype(dtype).itemsize > UNSLICED_LIMIT_BYTES


def needs_hop(left_env, right_env, ops, shape):
    if not USE_GPU or len(ops) not in (1, 2) or len(shape) != len(ops) + 2:
        return False
    dtype = xp.result_type(left_env, right_env, *ops)
    width = max(max(o.shape[0], o.shape[-1]) for o in ops)
    size = max(shape[0], left_env.shape[0]) * max(shape[-1], right_env.shape[0]) * width * int(np.prod(shape[1:-1]))
    return size * np.dtype(dtype).itemsize > UNSLICED_LIMIT_BYTES


def hop(left_env, right_env, ops, matrix, workspace=None):
    """One-/two-site H action, slicing the output right MPS index."""
    workspace = WORKSPACE_BYTES if workspace is None else workspace
    matrix = xp.asarray(matrix)
    dtype = xp.result_type(left_env, right_env, *ops, matrix)
    out = xp.empty((left_env.shape[0], *(o.shape[1] for o in ops), right_env.shape[0]), dtype=dtype)
    width = max(max(o.shape[0], o.shape[-1]) for o in ops)
    per = matrix.shape[0] * width * int(np.prod([max(o.shape[1:3]) for o in ops]))
    step = _chunk(right_env.shape[0], per, dtype, workspace)
    for start in range(0, right_env.shape[0], step):
        sl = slice(start, start + step)
        if len(ops) == 2:
            t = xp.einsum("cehk,ljk->cehlj", matrix, right_env[sl], optimize=True)
            t = xp.einsum("cehlj,fghj->celfg", t, ops[1], optimize=True)
            t = xp.einsum("celfg,bdef->clgbd", t, ops[0], optimize=True)
            out[..., sl] = xp.einsum("abc,clgbd->adgl", left_env, t, optimize=True)
        else:
            t = xp.einsum("cek,ljk->celj", matrix, right_env[sl], optimize=True)
            t = xp.einsum("celj,bdej->clbd", t, ops[0], optimize=True)
            out[..., sl] = xp.einsum("abc,clbd->adl", left_env, t, optimize=True)
    return out
