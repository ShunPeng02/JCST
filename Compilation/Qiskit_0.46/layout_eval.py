# @Time: 2025/8/31 19:08    @Author: P1zza
# import matplotlib
import os

import numpy.random

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import argparse
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import loads as qasm2_loads  # Qiskit>=0.46
from qiskit.qasm2 import load as qasm2_load  # Qiskit>=0.46

from qiskit.transpiler import CouplingMap, PassManager, Layout

# Built-in layout passes
from qiskit.transpiler.passes import TrivialLayout, SabreLayout, VF2Layout, DenseLayout, \
    ApplyLayout

from plot.plot_bar import plot_norm_bar
from plot.plot_cx_bar import plot_cx_bar
from plot.plot_cx_table import plot_cx_table
from plot.plot_depth_bar import plot_depth_bar
from utils import _raise_timeout

SEED_GLOBAL = 1234
random.seed(SEED_GLOBAL)
np.random.seed(SEED_GLOBAL)

from backends import *
from utils import *


@dataclass
class MethodCfg:
    key: str
    name: str


def layout_from_list(qc: QuantumCircuit, phys: List[int]) -> Layout:
    mapping = {qc.qubits[i]: phys[i] for i in range(qc.num_qubits)}
    return Layout(mapping)


def choose_initial_layout_builtin(qc: QuantumCircuit, cmap: CouplingMap, method_key: str, seed: int) -> Optional[
    Layout]:
    pm = PassManager()
    if method_key == "trivial":
        pm.append(TrivialLayout(cmap))
    elif method_key == "sabre":
        pm.append(SabreLayout(cmap, seed=seed))
    # elif method_key == "sabre_com":
    #     pm.append(SabrePreLayout(cmap))
    #     pm.append(SabreLayout(cmap, seed=seed))
    elif method_key == "vf2":
        pm.append(VF2Layout(cmap, seed=seed))
    elif method_key == "dense":
        pm.append(DenseLayout(cmap))
    elif method_key == "csp":
        pm.append(ApplyLayout())
    else:
        return None

    # 用pm跑layout
    _ = pm.run(qc)
    lay = pm.property_set.get("layout", None)
    if lay is None:
        return None
    phys: List[int] = []
    for q in qc.qubits:  # 遍历得到layout结果
        p = lay[q]
        phys.append(int(getattr(p, "index", p)))
    return layout_from_list(qc, phys)


def parse_args():
    p = argparse.ArgumentParser(description="B23 mapping-only benchmark on FakeTokyo (with ILS/TWP baselines)")
    p.add_argument("--seed", type=int, default=SEED_GLOBAL, help="Global seed")
    p.add_argument("--arch", type=str, default="FakeTokyo", choices=["Tokyo20", "Tokyo", "auckland"],
                   help="Use 20q subgraph or full FakeTokyo")
    # baseline_method = ["trivial", "sabre", "sabre_com", "vf2", "dense", "csp"]
    p.add_argument("--methods", type=str, nargs="+",
                   # TODO: NoiseAdaptiveLayout,解决模拟线路超时问题后，可以加入基线
                   default=["trivial", "sabre", "vf2", "dense"], help="Mapping methods to evaluate")
    p.add_argument("--qasm_path", type=str, default="qasm_examples/",
                   help="Base dir for B23 QASM files, e.g., 'B23/*.qasm'")
    p.add_argument("--out_path", type=str, default="results/b23_Tokyo20_mapping.csv", help="CSV output path")
    p.add_argument("--repeats", type=int, default=100, help="Repeats per circuit (different seeds)")
    p.add_argument("--limit", type=int, default=1, help="Limit number of circuits (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    # setting fake backend architecture
    args = parse_args()
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    backend = get_fake_tokyo()
    basis = backend_basis(backend)
    cmap = backend_cmap(backend)
    print(f"[INFO] Using full {args.arch} ({backend.configuration().num_qubits}).")

    # baseline config
    methods: List[MethodCfg] = []
    for k in args.methods:
        methods.append(MethodCfg(k, k.upper() if k in {"ils", "twp"} else k.capitalize() + "Layout"))
    if not methods:
        raise RuntimeError("No valid methods selected.")

    # load qasm files

    # init results csv file
    tabel_head = ["Circuit Name", "Qubit Num", "Gate Num", "method", "lay_ok",
                  "cx_ideal", "cx_constrained", "cx_added",
                  "depth_ideal", "depth_constrained",
                  "seed", "layout_time_s", "transpile_time_s", "size"]

    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(",".join(tabel_head) + "\n")

    # load .qasm & start mapping
    from qasm_files import SMAL_QASM_FILE

    for idx, path in enumerate(SMAL_QASM_FILE):
        qc = qasm2_load(args.qasm_path + path)
        qubits = n_active_qubits(qc)
        gate_num = sum(qc.count_ops().values())

        # ideal transpile
        try:

            t0 = time.perf_counter()
            tqc_ideal = transpile(
                circuits=qc,
                coupling_map=None,
                basis_gates=basis,
                optimization_level=0,
                routing_method=None,
                seed_transpiler=args.seed + idx,
            )
            cx_ideal = cx_count(tqc_ideal)
            depth_ideal = int(tqc_ideal.depth())
        except Exception as e:
            print(f"[WARN] Ideal transpile failed for {path.name}: {e}")
            cx_ideal = depth_ideal = -1

        # eval transpile
        for r in range(args.repeats):
            seed = args.seed + 100 * r + idx

            for m in methods:
                # initial layout (and its timing)
                tL0 = time.perf_counter()
                try:
                    init_layout = choose_initial_layout_builtin(qc, cmap, m.key, seed)
                    lay = "False" if init_layout is None else "True"
                    layout_time = time.perf_counter() - tL0
                except Exception as e:
                    print(f"[WARN] initial_layout failed ({m.key}) on {path}: {e}")
                    init_layout = None
                    layout_time = lay = -1.0

                # Constrained transpile under the target coupling map
                try:
                    timer = threading.Timer(60, lambda: _raise_timeout())  # 60 秒超时
                    timer.start()
                    t0 = time.perf_counter()
                    tqc = transpile(
                        qc,
                        coupling_map=cmap,
                        basis_gates=basis,
                        optimization_level=0,
                        initial_layout=init_layout,
                        routing_method="sabre",
                        seed_transpiler=seed,
                    )
                    # readout_error = get_tv_distance(tqc, backend)
                    trans_t = time.perf_counter() - t0
                    cx_c = cx_count(tqc)
                    dep = int(tqc.depth())
                    sz = int(tqc.size())
                    cx_added = max(0, cx_c - cx_ideal) if cx_ideal >= 0 else -1
                except Exception as e:
                    print(f"[ERROR] Constrained transpile failed ({path.name}, {m.key}): {e}")
                    trans_t = cx_c = dep = sz = cx_added = readout_error = -1
                except TimeoutError:
                    print(f"[ERROR] Transpile timed out after 60 seconds")
                    trans_t = cx_c = dep = sz = cx_added = readout_error = -1
                finally:
                    timer.cancel()  # 取消定时器，避免资源泄漏

                # layout失败 标记-1
                if init_layout is None:
                    cx_added = dep = -1
                row = [os.path.splitext(path)[0], qubits, gate_num, m.key, lay,
                       cx_ideal, cx_c, cx_added,
                       depth_ideal, dep,
                       seed, f"{layout_time:.6f}", f"{trans_t:.6f}", sz]
                with open(args.out_path, "a", encoding="utf-8") as f:
                    f.write(",".join(map(str, row)) + "\n")

    # Quick summary
    try:
        df = pd.read_csv(args.out_path)
        grp = (df.groupby(["arch", "method"])
               .agg(cx_added_mean=("cx_added", "mean"),
                    depth_mean=("depth", "mean"),
                    transpile_time_s=("transpile_time_s", "mean"))
               .reset_index())
        print("\n=== Summary over B23 (FakeTokyo) ===")
        print(grp.to_string(index=False))
        print(f"\nResults saved to {args.out_path}")
    except Exception:
        print(f"Results saved to {args.out_path}")

    plot_cx_table("results/b23_Tokyo20_mapping.csv", "results")
    # depth
    plot_norm_bar("depth", "mapping", "../results/b23_Tokyo20_mapping.csv", "../results")
    # cx
    plot_norm_bar("cx", "mapping", "../results/b23_Tokyo20_mapping.csv", "../results")
