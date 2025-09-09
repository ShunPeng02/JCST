# dj_qpanda.py
from pyqpanda import *


def build_dj_prog(n=3, kind="balanced_parity"):
    """
    kind: "constant_zero" | "constant_one" | "balanced_parity"
    """
    qvm = CPUQVM();
    qvm.init_qvm()  # 初始化模拟器 :contentReference[oaicite:0]{index=0}
    xs = qvm.qAlloc_many(n)  # 申请 n 个量子比特 :contentReference[oaicite:1]{index=1}
    anc = qvm.qAlloc()  # 辅助位
    cs = qvm.cAlloc_many(n)  # n 个经典位

    prog = QProg()

    # (A) 初态：|0>^n ⊗ |1> → |+>^n ⊗ |->
    prog << X(anc) << H(anc)
    for q in xs: prog << H(q)

    # prog << BARRIER(xs + [anc])           # U_f 结束（再加一条竖虚线）

    # (B) Oracle U_f
    for q in xs: prog << CNOT(q, anc)  # f(x)=x0⊕…⊕x_{n-1}
    # constant_zero: 不操作
    prog << BARRIER(xs + [anc])           # U_f 结束（再加一条竖虚线）


    # (C) 干涉+测量：只测输入寄存器
    for q in xs: prog << H(q)
    prog << measure_all(xs, cs)

    return qvm, prog, cs


def export_circuit(prog, stem="dj_qpanda", outdir="results"):
    import os
    os.makedirs(outdir, exist_ok=True)
    # 1) 直接出 PNG（最省事）
    draw_qprog(prog, 'pic', filename=f"{outdir}/{stem}.png")  # :contentReference[oaicite:2]{index=2}
    # 2) 导出 LaTeX（需要本机 LaTeX，得到 .tex）
    draw_qprog(prog, 'latex', filename=f"{outdir}/{stem}.tex")  # :contentReference[oaicite:3]{index=3}
    # 3) 也可打印 ASCII 文本图
    with open(f"{outdir}/{stem}.txt", "w", encoding="utf-8") as f:
        f.write(str(prog))  # 文本/ASCII 图 :contentReference[oaicite:4]{index=4}


if __name__ == "__main__":
    # constant: f(x)=0
    qvm, prog_c0, cs = build_dj_prog(n=3, kind="constant_zero")
    export_circuit(prog_c0, stem="DJ_qpanda")
