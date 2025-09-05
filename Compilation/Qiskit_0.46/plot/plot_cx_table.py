#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot_table2_simple.py  (new)
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_METHODS: List[str] = ["trivial", "sabre", "vf2", "dense"]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 要求必备列
    required = {
        "Circuit Name", "Qubit Num", "Gate Num",
        "method", "cx_added"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    return df


def build_pivot(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(["Circuit Name", "Qubit Num", "Gate Num", "method"], as_index=False)["cx_added"]
        .mean()
    )
    pv = agg.pivot(index=["Circuit Name", "Qubit Num", "Gate Num"],
                   columns="method", values="cx_added")
    pv = pv.reindex(columns=methods)

    pv_round = pv.round(0)

    # layout failed replace -1 with "-"
    pv_round = pv_round.replace(-1, "—")  # 或者 "–"

    out = pv_round.rename_axis(None, axis=1).reset_index()

    # sort by qubit num
    out = out.sort_values(by=["Qubit Num", "Circuit Name"], ascending=[True, True]).reset_index(drop=True)
    return out


def save_csv(df_out: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "cx_added_table.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def save_png(df_out: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "cx_added_table.png"

    # 简单自适应尺寸
    n_rows, n_cols = df_out.shape
    fig_w = max(8.0, 1.2 * n_cols)
    fig_h = max(6.0, 0.35 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_out.astype(object).values.tolist(),
        colLabels=df_out.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    # === 在这里加标题 ===
    ax.set_title("Number of additional CNOT gates", fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_cx_table(IN_PATH, OUT_PATH):
    df = load_results(IN_PATH)
    pv = build_pivot(df, methods=DEFAULT_METHODS)
    out_csv = save_csv(pv, Path(OUT_PATH))
    # out_png = save_png(pv, OUT_DIR)
    print(f"[OK] Wrote: {out_csv}")
    # print(f"[OK] Wrote: {out_png}")
    return
