#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_METHODS: List[str] = ["trivial", "sabre", "vf2", "dense"]


def load_and_aggregate(csv_path: str, methods: List[str]) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    df = df[df["method"].isin(methods)].copy()

    # 先在 (Circuit Name, method) 维度上合并多行（不同seed/repeat），取平均
    df_agg = (
        df.groupby(["Circuit Name", "method"], as_index=False)["read_error"]
          .mean()
          .rename(columns={"read_error": "read_error_mean"})
    )

    # 为后续画图方便，也保留“每方法一个列表”的展开形式
    return df_agg


def plot_bar(df_agg: pd.DataFrame, methods: List[str], title: str, out_png: Path):
    """
    每个方法的平均 read_error（跨Circuit再求 mean），并加 std 误差条。
    """
    stats = (
        df_agg.groupby("method")["read_error_mean"]
              .agg(["mean", "std", "count"])
              .reindex(methods)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stats))
    ax.bar(x, stats["mean"].values, yerr=stats["std"].values, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index.tolist(), rotation=0)
    ax.set_ylabel("Mean readout/output error (TV distance)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_box(df_agg: pd.DataFrame, methods: List[str], title: str, out_png: Path):
    """
    每个方法一个箱线图，展示各电路上的 read_error_mean 分布（更贴近Fig.16(a)“看分布”的风格）。
    """
    data = [df_agg[df_agg["method"] == m]["read_error_mean"].values for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=methods, showfliers=True)
    ax.set_ylabel("Readout/output error (TV distance)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():


    out_dir = Path("results/")

    df_agg = load_and_aggregate("results/b23_tokyo.csv", DEFAULT_METHODS)

    # agg_csv = out_dir / "read_error_agg.csv"
    # df_agg.to_csv(agg_csv, index=False, encoding="utf-8")
    # print(f"[OK] 写出聚合明细: {agg_csv}")

    bar_png = out_dir / "read_error_bar.png"
    plot_bar(df_agg, DEFAULT_METHODS, "read out error", bar_png)
    print(f"[OK] bar plot saved: {bar_png}")

    # box_png = out_dir / "read_error_box.png"
    # plot_box(df_agg, args.methods, args.title, box_png)
    # print(f"[OK] 写出箱线图: {box_png}")


if __name__ == "__main__":
    main()
