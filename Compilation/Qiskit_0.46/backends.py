# @Time: 2025/8/31 16:14    @Author: P1zza
from typing import List

from qiskit import transpile, QuantumCircuit
from qiskit.transpiler import CouplingMap


def get_fake_tokyo():
    from qiskit.providers.fake_provider import FakeTokyo
    return FakeTokyo()


def get_fake_auckland():
    from qiskit.providers.fake_provider import FakeAuckland
    return FakeAuckland()


def backend_basis(backend) -> List[str]:
    try:
        return list(backend.configuration().basis_gates)
    except Exception:
        try:
            return sorted(list(backend.target.operation_names))
        except Exception:
            return ["rz", "sx", "x", "cx"]


def backend_cmap(backend) -> CouplingMap:
    return CouplingMap(backend.configuration().coupling_map)

