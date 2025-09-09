# qiskit_circuit_viz.py
# Qiskit >= 0.44 (Terra) 兼容；仅用于生成“电路图”，无需执行仿真
from pathlib import Path
from typing import List, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector



def save_circuit_diagram(circ: QuantumCircuit, fname: str, outdir: Path, output: str = "mpl", scale: float = 1.2):
    outdir.mkdir(parents=True, exist_ok=True)
    path_png = outdir / f"{fname}_{output}.png"
    path_pdf = outdir / f"{fname}_{output}.pdf"
    # 注意：fold=-1 避免自动换行，方便横向对比
    circ.draw(output=output, fold=-1, scale=scale, filename=str(path_png), style="iqp")

    # 再另存一份 PDF（有的 Qiskit 版本不支持直接 filename=PDF，这里简单再画一次）
    circ.draw(output=output, fold=-1, scale=scale, filename=str(path_pdf), style="iqp")
    print(f"[saved] {path_png}")
    print(f"[saved] {path_pdf}")


# =========================
# Deutsch–Jozsa
# =========================
def deutsch_jozsa_oracle(n: int, balanced: bool) -> QuantumCircuit:
    """
    构造 n 比特输入 + 1 比特辅助位 的 DJ oracle。
    - constant: f(x)=0（恒等），或 f(x)=1（整体在辅助位上加个X）
    - balanced: 这里构造一个简单的线性平衡函数：f(x)=x_0 XOR x_1 XOR ... XOR x_{n-1}
    """
    oracle = QuantumCircuit(n + 1, name="U_f")
    if balanced:
        # 对每个输入位添加 CNOT 到辅助位，实现 parity（奇偶）函数（平衡）
        for i in range(n):
            oracle.cx(i, n)
    else:
        # constant：保持不变（f(x)=0），也可改成整体在辅助位加 X（f(x)=1）
        # 这里选择 f(x)=0，即空操作
        pass
    return oracle


def deutsch_jozsa_circuit(n: int, balanced: bool) -> QuantumCircuit:
    """
    Deutsch–Jozsa 主电路：
    - n 个输入位初始化 |0⟩
    - 1 个辅助位初始化 |1⟩（先X后H）
    - 对所有比特施加 H
    - 应用 oracle
    - 对前 n 个输入位施加 H 并测量
    """
    x = QuantumRegister(n, "x")
    anc = QuantumRegister(1, "anc")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(x, anc, c, name=f"DJ_{'balanced' if balanced else 'constant'}")

    # 初始化辅助位到 |1⟩ 并 H；输入位先不动，后一起 H
    qc.x(anc)
    qc.h(anc)
    qc.h(x)

    # Oracle
    U_f = deutsch_jozsa_oracle(n, balanced=balanced)
    qc.append(U_f.to_gate(), [*x, anc[0]])

    # 反变换并测量输入位
    qc.h(x)
    qc.measure(x, c)
    return qc


if __name__ == "__main__":
    outdir = Path("results")

    # ---- Deutsch–Jozsa：n=3，分别生成 constant / balanced ----
    dj_const = deutsch_jozsa_circuit(n=3, balanced=True)

    save_circuit_diagram(dj_const, "DJ_qiskit", outdir, output="mpl", scale=1.3)

    save_circuit_diagram(dj_const, "DJ_qiskit", outdir, output="latex", scale=1.3)



