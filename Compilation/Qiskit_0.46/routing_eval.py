# -*- coding: utf-8 -*-
# nohup python3 routing_eval.py > routing_eval.output 2>&1 &

import os
import signal
import time
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import load as qasm2_load  # Qiskit>=0.46
from qiskit.transpiler import CouplingMap, PassManager, Layout, TransformationPass
from qiskit.transpiler.passes import SabreLayout, ApplyLayout, BasicSwap, LookaheadSwap, StochasticSwap, SabreSwap

from backends import *
from plot.plot_bar import plot_norm_bar
from utils import *
from utils import _raise_timeout

SEED_GLOBAL = 1234
random.seed(SEED_GLOBAL)
np.random.seed(SEED_GLOBAL)


@dataclass
class MethodCfg:
    key: str   # e.g., "sabre"
    name: str  # e.g., "SabreRouting" (for display if needed)


def layout_from_list(qc: QuantumCircuit, phys: List[int]) -> Layout:
    mapping = {qc.qubits[i]: phys[i] for i in range(qc.num_qubits)}
    return Layout(mapping)


def sabre_initial_layout(qc: QuantumCircuit, cmap: CouplingMap, seed: int) -> Optional[Layout]:
    """Return a Layout computed by SabreLayout for a fixed initial mapping."""
    pm = PassManager()
    pm.append(SabreLayout(cmap, seed=seed))
    _ = pm.run(qc)
    lay = pm.property_set.get("layout", None)
    if lay is None:
        return None
    phys: List[int] = []
    for q in qc.qubits:
        p = lay[q]
        phys.append(int(getattr(p, "index", p)))
    return layout_from_list(qc, phys)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="B23 routing-only benchmark on FakeTokyo (fixed initial layout=Sabre)")
    p.add_argument("--seed", type=int, default=SEED_GLOBAL, help="Global seed")
    p.add_argument("--arch", type=str, default="FakeTokyo", choices=["Tokyo20", "Tokyo", "auckland"],
                   help="Use 20q subgraph or full FakeTokyo")
    # Supported routing methods in Qiskit: basic, lookahead, stochastic, sabre; 'commuting' if available.
    # ['basic', 'lookahead', 'stochastic', 'sabre']:
    p.add_argument("--methods", type=str, nargs="+",
                   default=['basic', 'lookahead', 'stochastic', 'sabre'],
                   help="Routing methods to evaluate")
    p.add_argument("--qasm_path", type=str, default="qasm_examples/",
                   help="Base dir for B23 QASM files, e.g., 'qasm_examples/'")
    p.add_argument("--out_path", type=str, default="results/b23_Tokyo20_routing.csv", help="CSV output path")
    p.add_argument("--repeats", type=int, default=10, help="Repeats per circuit (different seeds)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of circuits (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = get_fake_tokyo()
    basis = backend_basis(backend)
    cmap = backend_cmap(backend)
    print(f"[INFO] Using full {args.arch} ({backend.configuration().num_qubits}).")

    # Method configs
    methods: List[MethodCfg] = [MethodCfg(k, k.capitalize() + "Routing") for k in args.methods]
    if not methods:
        raise RuntimeError("No routing methods selected.")

    # CSV header
    table_head = ["Circuit Name", "Qubit Num", "Gate Num", "method", "roting_ok",
                  "cx_ideal", "cx_constrained", "cx_added", "depth_ideal", "depth_constrained",
                  "seed", "layout_time_s", "transpile_time_s", "size"]


    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(",".join(table_head) + "\n")

    # Load QASM file list
    from qasm_files import SMAL_QASM_FILE
    file_list = SMAL_QASM_FILE
    if args.limit and args.limit > 0:
        file_list = file_list[:args.limit]

    for idx, path in enumerate(file_list):
        qc = qasm2_load(args.qasm_path + path)
        qubits = n_active_qubits(qc)
        gate_num = sum(qc.count_ops().values())

        # Ideal (fully-connected) compile to get cx_ideal & depth_ideal
        try:
            t0 = time.perf_counter()
            tqc_ideal = transpile(
                circuits=qc,
                coupling_map=None,            # fully connected
                basis_gates=basis,            # expected ["cx", "rz", "sx", "x"]
                optimization_level=0,
                routing_method=None,
                seed_transpiler=args.seed,
            )
            cx_ideal = cx_count(tqc_ideal)
            depth_ideal = int(tqc_ideal.depth())
        except Exception as e:
            print(f"[WARN] Ideal transpile failed for {path}: {e}")
            cx_ideal = depth_ideal = -1

        # Repeats over seeds & methods
        for r in range(args.repeats):
            time_out = 10
            seed = args.seed + 1 * r
            # seed = args.seed

            # Fixed initial layout: SabreLayout
            tL0 = time.perf_counter()
            try:
                init_layout = sabre_initial_layout(qc, cmap, seed)
                layout_time = time.perf_counter() - tL0
            except Exception as e:
                print(f"[WARN] Sabre initial layout failed on {path}: {e}")
                init_layout = None
                layout_time = -1.0

            # Constrained transpile with selected routing method
            for m in methods:
                try:              # 注册超时信号，仅在“受限编译”阶段启用
                    signal.signal(signal.SIGALRM, _raise_timeout)
                    signal.alarm(time_out)

                    t0 = time.perf_counter()
                    tqc = transpile(
                        qc,
                        coupling_map=cmap,
                        basis_gates=basis,
                        optimization_level=0,
                        initial_layout=init_layout,
                        routing_method=m.key,
                        # seed_transpiler=seed,
                    )
                    # readout_error = get_tv_distance(tqc, backend)
                    trans_t = time.perf_counter() - t0
                    cx_c = cx_count(tqc)
                    dep = int(tqc.depth())
                    sz = int(tqc.size())
                    cx_added = max(0, cx_c - cx_ideal) if cx_ideal >= 0 else -1
                    routing_ok = 1
                except Exception as e:
                    # 包含 TimeoutError 在内的任何异常都记为失败，但继续后续任务
                    signal.alarm(0)  # 异常也要清除闹钟，避免影响后续循环
                    print(f"[ERROR] Constrained transpile failed ({path}, {m.key}): {e}: {time_out}")
                    trans_t = cx_c = dep = sz = cx_added = routing_ok = -1


                row = [os.path.splitext(path)[0], qubits, gate_num, m.key, str(routing_ok),
                       cx_ideal, cx_c, cx_added, depth_ideal, dep,
                       seed, f"{layout_time:.6f}", f"{trans_t:.6f}", sz]

                with open(args.out_path, "a", encoding="utf-8") as f:
                    f.write(",".join(map(str, row)) + "\n")

    # Quick summary
    try:
        df = pd.read_csv(args.out_path)
        grp = (df.groupby(["method"])
               .agg(cx_added_mean=("cx_added", "mean"),
                    depth_constrained_mean=("depth_constrained", "mean"),
                    transpile_time_mean=("transpile_time_s", "mean"))
               .reset_index())
        print("\n=== Summary over B23 (FakeTokyo) ===")
        print(grp.to_string(index=False))
        print(f"\nResults saved to {args.out_path}")
    except Exception:
        print(f"Results saved to {args.out_path}")

    # depth
    plot_norm_bar("depth", "routing", "../results/b23_Tokyo20_routing.csv", "../results")
    # cx
    plot_norm_bar("cx", "routing", "../results/b23_Tokyo20_routing.csv", "../results")
