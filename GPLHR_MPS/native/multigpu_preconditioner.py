from __future__ import annotations

import argparse
import os
from multiprocessing.connection import Connection
import socket
import subprocess
import sys
import time
import traceback
from typing import Sequence

import numpy as np


def native_mps_to_payload(state) -> tuple:
    """Move only allowed native MPS blocks to a picklable CPU payload."""

    from renormalizer.mps.matrix import asnumpy

    site_blocks = [
        {
            key: np.asarray(asnumpy(value), dtype=np.complex128).copy()
            for key, value in blocks.items()
        }
        for blocks in state.site_blocks
    ]
    return (
        site_blocks,
        [dict(dimensions) for dimensions in state.bond_dims],
        tuple(state.total_qn),
    )


def payload_to_native_mps(payload):
    """Reconstruct a symmetry-native MPS on the process's current device."""

    from .native_tensor import NativeMPS

    site_blocks, bond_dims, total_qn = payload
    state = NativeMPS(
        [
            {key: np.asarray(value).copy() for key, value in blocks.items()}
            for blocks in site_blocks
        ],
        [dict(dimensions) for dimensions in bond_dims],
        tuple(total_qn),
    )
    return state


def _balanced_groups(columns: int, devices: int) -> list[list[int]]:
    """Assign the extra columns to the last devices to isolate root zero."""

    groups = [[] for _ in range(devices)]
    initial = min(columns, devices)
    for column in range(initial):
        groups[column].append(column)
    device = devices - 1
    for column in range(initial, columns):
        groups[device].append(column)
        device -= 1
        if device < 1:
            device = devices - 1
    return groups


class MultiGPUOneSitePreconditioner:
    """Persistent process pool for a block of one-site preconditioners."""

    def __init__(self, parameters, config, local_preconditioner, devices: int):
        from renormalizer.mps.backend import USE_GPU, xp

        if not USE_GPU:
            raise RuntimeError("multi-GPU preconditioning requires CUDA")
        available = int(xp.cuda.runtime.getDeviceCount())
        if not 2 <= devices <= available:
            raise ValueError(
                f"requested {devices} preconditioner GPUs, CUDA exposes {available}"
            )
        self.parameters = parameters
        self.config = config
        self.local_preconditioner = local_preconditioner
        self.devices = int(devices)
        self.processes: list[subprocess.Popen] = []
        self.connections: list[Connection] = []
        self.calls = 0
        self.columns = 0
        self.cpu_transfer_seconds = 0.0
        self.wall_seconds = 0.0
        self._closed = False

        for device in range(1, self.devices):
            parent_socket, child_socket = socket.socketpair()
            environment = os.environ.copy()
            environment["RENO_GPU"] = str(device)
            environment["GPLHR_GPU_STREAMS"] = "1"
            command = [
                sys.executable,
                "-u",
                "-m",
                (
                    "GPLHR_MPS.native."
                    "multigpu_preconditioner"
                ),
                "--worker-fd",
                str(child_socket.fileno()),
                "--device",
                str(device),
            ]
            process = subprocess.Popen(
                command,
                env=environment,
                pass_fds=(child_socket.fileno(),),
                close_fds=True,
            )
            child_socket.close()
            connection = Connection(parent_socket.detach())
            ready = connection.recv()
            if ready != {"status": "ready", "device": device}:
                process.terminate()
                raise RuntimeError(
                    f"multi-GPU worker {device} did not initialize: {ready!r}"
                )
            connection.send(
                {
                    "command": "initialize",
                    "parameters": parameters,
                    "config": config,
                }
            )
            initialized = connection.recv()
            if initialized != {"status": "initialized", "device": device}:
                process.terminate()
                raise RuntimeError(
                    f"multi-GPU worker {device} rejected configuration: "
                    f"{initialized!r}"
                )
            self.processes.append(process)
            self.connections.append(connection)

    def apply_block(
        self,
        states: Sequence[object],
        call_indices: Sequence[int],
    ) -> list[tuple[object, dict, float]]:
        if len(states) != len(call_indices):
            raise ValueError("preconditioner states and call indices differ")
        if not states:
            return []
        started = time.perf_counter()
        groups = _balanced_groups(len(states), self.devices)
        transfer_started = time.perf_counter()
        payloads = [None] * len(states)
        for device in range(1, self.devices):
            for column in groups[device]:
                payloads[column] = native_mps_to_payload(states[column])
        self.cpu_transfer_seconds += time.perf_counter() - transfer_started

        for device in range(1, self.devices):
            tasks = [
                {
                    "column": column,
                    "call_index": int(call_indices[column]),
                    "state": payloads[column],
                }
                for column in groups[device]
            ]
            self.connections[device - 1].send(
                {"command": "apply_block", "tasks": tasks}
            )

        results: list[tuple[object, dict, float] | None] = [None] * len(states)
        for column in groups[0]:
            column_started = time.perf_counter()
            output, report = self.local_preconditioner.apply(
                states[column], call_index=int(call_indices[column])
            )
            report = {
                **report,
                "parallel_worker_device": 0,
                "multi_gpu_transfer": "none; coordinator-local column",
            }
            results[column] = (
                output,
                report,
                time.perf_counter() - column_started,
            )

        transfer_started = time.perf_counter()
        for device in range(1, self.devices):
            message = self.connections[device - 1].recv()
            if message.get("status") != "ok":
                raise RuntimeError(
                    f"multi-GPU preconditioner worker {device} failed:\n"
                    f"{message.get('traceback', message)!s}"
                )
            for item in message["results"]:
                column = int(item["column"])
                output = payload_to_native_mps(item["state"])
                output.validate(self.config.max_bond)
                report = {
                    **item["report"],
                    "parallel_worker_device": device,
                    "multi_gpu_transfer": (
                        "native symmetry blocks via CPU; no sector vector"
                    ),
                    "worker_task_seconds": float(item["seconds"]),
                }
                results[column] = (output, report, float(item["seconds"]))
        self.cpu_transfer_seconds += time.perf_counter() - transfer_started

        if any(result is None for result in results):
            raise RuntimeError("multi-GPU preconditioner lost a block column")
        self.calls += 1
        self.columns += len(states)
        self.wall_seconds += time.perf_counter() - started
        return list(results)  # type: ignore[arg-type]

    def audit(self) -> dict:
        return {
            "enabled": True,
            "execution": "persistent process-isolated CUDA workers",
            "devices": self.devices,
            "coordinator_device": 0,
            "worker_devices": list(range(1, self.devices)),
            "calls": self.calls,
            "columns": self.columns,
            "wall_seconds": self.wall_seconds,
            "cpu_symmetry_block_transfer_seconds": self.cpu_transfer_seconds,
            "transferred_representation": (
                "allowed native MPS tensor blocks and QN bond maps only"
            ),
            "full_hilbert_vectors": 0,
            "fixed_sector_vectors": 0,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _worker_main(worker_fd: int, device: int) -> None:
    connection = Connection(worker_fd)
    connection.send({"status": "ready", "device": device})
    initialization = connection.recv()
    if initialization.get("command") != "initialize":
        raise RuntimeError("worker did not receive initialization")

    from .native_gplhr import _NativeOneSiteGMRESPreconditioner

    config = initialization["config"]
    preconditioner = _NativeOneSiteGMRESPreconditioner(
        initialization["parameters"], config
    )
    connection.send({"status": "initialized", "device": device})

    while True:
        message = connection.recv()
        command = message.get("command")
        if command == "close":
            connection.close()
            return
        if command != "apply_block":
            raise RuntimeError(f"unknown worker command {command!r}")
        try:
            output_rows = []
            for task in message["tasks"]:
                started = time.perf_counter()
                state = payload_to_native_mps(task["state"])
                state.validate(config.max_bond)
                output, report = preconditioner.apply(
                    state, call_index=int(task["call_index"])
                )
                output.validate(config.max_bond)
                output_rows.append(
                    {
                        "column": int(task["column"]),
                        "state": native_mps_to_payload(output),
                        "report": report,
                        "seconds": time.perf_counter() - started,
                    }
                )
            connection.send({"status": "ok", "results": output_rows})
        except Exception:
            connection.send(
                {"status": "error", "traceback": traceback.format_exc()}
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-fd", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    _worker_main(args.worker_fd, args.device)


if __name__ == "__main__":
    main()


__all__ = [
    "MultiGPUOneSitePreconditioner",
    "native_mps_to_payload",
    "payload_to_native_mps",
]
