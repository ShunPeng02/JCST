#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import subprocess

from projectq import MainEngine
from projectq.backends import CircuitDrawer
from projectq.ops import H, X, CNOT, All, Measure
from projectq.ops import BasicGate


class UfGate(BasicGate):
    def __str__(self):  return "U_f"

    def tex_str(self):  return r"\mathrm{U_f}"


def build_dj_projectq(n: int = 3, balanced: bool = True) -> CircuitDrawer:
    """
    Deutsch–Jozsa：balanced=False -> constant(f(x)=0)
                   balanced=True  -> parity-balanced(f(x)=x0 ⊕ ... ⊕ x_{n-1})
    """
    drawer = CircuitDrawer()
    eng = MainEngine(backend=drawer)

    xs = eng.allocate_qureg(n)
    anc = eng.allocate_qubit()

    # (A) 初态：|0>^n ⊗ |1>，再到 |+>^n ⊗ |->
    X | anc  # 改：单比特不要用 All
    H | anc  # 改：单比特不要用 All
    All(H) | xs
    eng.flush()  # ← 强制“切断”，前一层 H 会被画出来

    # (B) Oracle：parity-balanced 用 CNOT 链；constant 什么都不做
    if balanced:
        for q in xs:
            CNOT | (q, anc)
    else:
        UfGate() | tuple(xs + [anc])


    eng.flush()  # ← 强制“切断”，前一层 H 会被画出来

    # (C) 干涉读取：仅对输入位再做 H，并测量输入位（方便在图里显示测量）
    All(H) | xs
    All(Measure) | xs  # 还原测量，电路图才会画出测量符号

    eng.flush()
    return drawer


if __name__ == "__main__":
    n = 3
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)  # 新增：确保目录存在

    # constant: f(x)=0
    drawer_const = build_dj_projectq(n=n, balanced=True)
    with open("results/DJ_ProjectQ.tex", 'w') as f:
        f.write(drawer_const.get_latex())

    # balanced: f(x)=x0 ⊕ ...（parity）
    # drawer_bal = build_dj_projectq(n=n, balanced=True)
    # (outdir / f"DJ_ProjectQ_balanced_n{n}.tex").write_text(
    #     drawer_bal.get_latex(), encoding="utf-8"
    # )

    # 可选：自动编译为 PDF（若存在 pdflatex）
    # pdflatex = shutil.which("pdflatex")
    # if pdflatex:
    #     for stem in [f"DJ_ProjectQ_constant_n{n}"]:
    #         subprocess.run(
    #             [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
    #              "-output-directory", str(outdir), str(outdir / f"{stem}.tex")],
    #             check=True
    #         )
    #     print("[OK] 已生成 PDF 到:", outdir.resolve())
    # else:
    #     print("[hint] 未检测到 pdflatex，仅生成 .tex。")
