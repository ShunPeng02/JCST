from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cirq
import sympy as sp
from cirq.contrib import circuit_to_latex_using_qcircuit

try:
    from cirq.contrib.svg import SVGCircuit, circuit_to_svg  # 官方 SVG 电路渲染
except Exception as e:  # pragma: no cover
    raise ImportError("需要 cirq.contrib.svg：请确保安装 Cirq >= 1.3") from e


class UfConstZero(cirq.Gate):
    def __init__(self, n_inputs: int, label: str = "U_f"):
        self.n, self.label = n_inputs, label

    def _num_qubits_(self): return self.n + 1

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=tuple([self.label] * (self.n + 1)))
    # 不需要分解：恒等门


def build_dj_circuit(n=3, balanced=False):
    xs = cirq.LineQubit.range(n)
    anc = cirq.NamedQubit("anc")
    c = cirq.Circuit()

    # (A) 初态
    c.append([cirq.X(anc)])
    c.append(cirq.H.on_each(*(xs + [anc])))

    # (B) U_f
    if balanced:
        for q in xs:
            c.append(cirq.CNOT(q, anc))
    else:
        gate = UfConstZero(n)
        c.append(gate.on(*xs, anc))


    # (C) 干涉+测量：只测输入寄存器
    c.append(cirq.H.on_each(*xs))
    c.append(cirq.measure(*xs, key="m"))
    return c


if __name__ == "__main__":
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Deutsch–Jozsa：n=3，constant / balanced ----
    dj_const = build_dj_circuit(n=3, balanced=True)
    print(dj_const)
    svg = circuit_to_svg(dj_const)
    with open("results/DJ_Cirq.svg", "w") as f:
        f.write(svg)

    with open("results/DJ_Cirq.png", "w") as f:
        f.write(svg)

    # TODO: latex file to pdf
    # with open("dj_Cirq.tex", "w") as f:
    #     f.write(circuit_to_latex_using_qcircuit(dj_const))

    # from cirq.contrib.svg import circuit_to_svg
    # import cairosvg
    #
    # cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(outdir / "dj_const.pdf"))
    # cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(outdir / "dj_const.png"))
