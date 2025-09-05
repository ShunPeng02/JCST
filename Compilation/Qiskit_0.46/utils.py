# @Time: 2025/9/1 15:40    @Author: P1zza
from qiskit import QuantumCircuit

import threading
import time


def timeout_handler(signum, frame):
    raise TimeoutError("Transpile operation exceeded 60 seconds")


def _raise_timeout(signum, frame):
    """Signal handler that raises a TimeoutError when an alarm fires."""
    raise TimeoutError("Operation timed out")


def n_active_qubits(qc: QuantumCircuit) -> int:
    """Count qubits that actually appear in any gate (some B23 files declare 16 but use fewer)."""
    active = set()
    for inst, qargs, _ in qc.data:
        for q in qargs:
            # robustly get integer index
            try:
                active.add(q.index)
            except AttributeError:
                active.add(qc.find_bit(q).index)
    return len(active)


def cx_count(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    return int(ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0))


def u1q_count(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    return int(ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0))


def cx_layers(qc: QuantumCircuit) -> int:
    try:
        return int(qc.depth(lambda nd: getattr(getattr(nd, "op", None), "name", "") in {"cx", "ecr", "cz"}))
    except Exception:
        return -1
