"""Thin torch.distributed wrapper for multi-process inference.

Inference is pure data parallelism (every rank computes its own shard; a single barrier is needed
only before merging) and requires no NCCL collectives -- so the **gloo** backend is used: CPU/TCP is
enough to synchronise across machines, and there is no NCCL watchdog timeout killing processes
(uneven per-rank inference time is the norm).

In a single process (WORLD_SIZE unset or =1) every function degrades to a no-op: the CPU smoke test /
single-GPU path behaves unchanged.
"""
import datetime
import os

import torch.distributed as dist


def init_distributed():
    """Initialise the gloo process group under torchrun multi-processing; returns (rank, world_size, local_rank)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("gloo", timeout=datetime.timedelta(hours=12))
    return rank, world, local_rank


def dist_info():
    """(rank, world_size); (0, 1) for a single process when no process group is initialised."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
