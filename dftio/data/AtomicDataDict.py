"""nequip.data.jit: TorchScript functions for dealing with AtomicData.

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Authors: Zhanghao Zhouyin zhouyinzhanghao@gmail.com
mofified from code by: Albert Musaelian
"""
from typing import Dict, Any

import numpy as np

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

# Define a type alias
Type = Dict[str, np.ndarray]

def validate_keys(keys, graph_required=True):
    # Validate combinations
    if graph_required:
        if not (_keys.POSITIONS_KEY in keys and _keys.EDGE_INDEX_KEY in keys):
            raise KeyError("At least pos and edge_index must be supplied")
    if _keys.EDGE_CELL_SHIFT_KEY in keys and "cell" not in keys:
        raise ValueError("If `edge_cell_shift` given, `cell` must be given.")


def with_edge_vectors(data: Type, with_lengths: bool = True) -> Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.EDGE_VECTORS_KEY in data:
        if with_lengths and _keys.EDGE_LENGTH_KEY not in data:
            data[_keys.EDGE_LENGTH_KEY] = np.linalg.norm(
                data[_keys.EDGE_VECTORS_KEY], axis=-1
            )

        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        if _keys.CELL_KEY in data and np.sum(np.abs(data[_keys.CELL_KEY])) > 1e-10:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[_keys.CELL_KEY].reshape(-1, 3, 3)
            edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
            if cell.shape[0] > 1:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                edge_vec = edge_vec + np.einsum(
                    "ni,nij->nj", edge_cell_shift, cell[batch[edge_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                edge_vec = edge_vec + np.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )

        data[_keys.EDGE_VECTORS_KEY] = edge_vec
        if with_lengths:
            data[_keys.EDGE_LENGTH_KEY] = np.linalg.norm(edge_vec, axis=-1)
        return data

def with_env_vectors(data: Type, with_lengths: bool = True) -> Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.ENV_VECTORS_KEY in data:
        if with_lengths and _keys.ENV_LENGTH_KEY not in data:
            data[_keys.ENV_LENGTH_KEY] = np.linalg.norm(
                data[_keys.ENV_VECTORS_KEY], axis=-1
            )
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        env_index = data[_keys.ENV_INDEX_KEY]
        env_vec = pos[env_index[1]] - pos[env_index[0]]
        if _keys.CELL_KEY in data and np.sum(np.abs(data[_keys.CELL_KEY])) > 1e-10:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[_keys.CELL_KEY].reshape(-1, 3, 3)
            env_cell_shift = data[_keys.ENV_CELL_SHIFT_KEY]
            if cell.shape[0] > 1:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                env_vec = env_vec + np.einsum(
                    "ni,nij->nj", env_cell_shift, cell[batch[env_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                env_vec = env_vec + np.einsum(
                    "ni,ij->nj",
                    env_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[_keys.ENV_VECTORS_KEY] = env_vec
        if with_lengths:
            data[_keys.ENV_LENGTH_KEY] = np.linalg.norm(env_vec, dim=-1)
        return data
    
def with_onsitenv_vectors(data: Type, with_lengths: bool = True) -> Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.ONSITENV_VECTORS_KEY in data:
        if with_lengths and _keys.ONSITENV_LENGTH_KEY not in data:
            data[_keys.ONSITENV_LENGTH_KEY] = np.linalg.norm(
                data[_keys.ONSITENV_VECTORS_KEY], axis=-1
            )
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        env_index = data[_keys.ONSITENV_INDEX_KEY]
        env_vec = pos[env_index[1]] - pos[env_index[0]]
        if _keys.CELL_KEY in data and np.sum(np.abs(data[_keys.CELL_KEY])) > 1e-10:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[_keys.CELL_KEY].reshape(-1, 3, 3)
            env_cell_shift = data[_keys.ONSITENV_CELL_SHIFT_KEY]
            if cell.shape[0] > 1:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                env_vec = env_vec + np.einsum(
                    "ni,nij->nj", env_cell_shift, cell[batch[env_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                env_vec = env_vec + np.einsum(
                    "ni,ij->nj",
                    env_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[_keys.ONSITENV_VECTORS_KEY] = env_vec
        if with_lengths:
            data[_keys.ONSITENV_LENGTH_KEY] = np.linalg.norm(env_vec, dim=-1)
        return data


def with_batch(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        return data
    else:
        pos = data[_keys.POSITIONS_KEY]
        batch = np.zeros(len(pos), dtype=np.int64)
        data[_keys.BATCH_KEY] = batch
        # ugly way to make a tensor of [0, len(pos)], but it avoids transfers or casts
        data[_keys.BATCH_PTR_KEY] = np.arange(
            start=0,
            stop=len(pos) + 1,
            step=len(pos),
            dtype=np.int64
        )
        
        return data
