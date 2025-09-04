import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== Your updated palette (from your modified file) =====
PALETTE = {
    "QFAST": "#B9B7B3",  # Ideal neutral gray
    "QuCT": "#4865A9",
    "CCD": "#EF8A43",
    "QSD": "#D9423C",
    "Squander": "#854C98",
}


def _format_number(v, metric):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if metric == "time_sec":
        v = float(v)
        if v >= 86400:  return f"{v / 86400:.1f}d"
        if v >= 3600:   return f"{v / 3600:.1f}h"
        if v >= 60:     return f"{v / 60:.1f}m"
        return f"{v:.1f}s"
    v = float(v)
    if v >= 1e6: return f"{v / 1e6:.1f}M"
    if v >= 1e3: return f"{v / 1e3:.1f}K"
    if v >= 10:  return f"{v:.1f}"
    return f"{v:.2f}"


def plot_quct_bar(
        in_path: str = "../results/synthesis_table.csv",
        out_path: str = "../results",
        metric: str = "gate",  # "gate" | "depth" | "time" | "time_sec"
        cols: tuple[str, ...] = ("random_4q", "random_5q", "benchmark_4q", "benchmark_5q"),
        col_labels: tuple[str, ...] = ("Ran.4q", "Ran.5q", "Bench.4q", "Bench.5q"),
        methods: tuple[str, ...] = ("QFAST", "QuCT", "CCD", "QSD", "Squander"),
        sqrt_scale: bool = True,
):
    """Draw grouped normalized bars (QFAST=1.0) and save PNG.

    Args:
        in_path: 输入CSV路径（例如 "../results/synthesis_table.csv" 或 "/mnt/data/quct_table4_numeric.csv"）
        metric: "gate" / "depth" / "time"（自动映射到 time_sec）/ "time_sec"
        out_path: 输出PNG路径；默认为 "../results/quct_<metric>_bar.png"
        cols: 要绘制的列键集合（默认去掉8q）
        col_labels: 横轴显示名称（与 cols 对应）
        methods: 方法顺序（需与CSV中的 method 名一致且包含 "QFAST"）
        sqrt_scale: 是否使用√纵轴
    """
    in_path = Path(in_path)
    df = pd.read_csv(in_path)

    sel_metric = metric if metric != "time" else "time_sec"
    ylabel = {"gate": "Gate Count", "depth": "Depth", "time_sec": "Time (s)"}.get(sel_metric, sel_metric)
    title_suffix = {"gate": "Gate", "depth": "Depth", "time_sec": "Time"}.get(sel_metric, sel_metric)

    # 取绝对值矩阵 (method x cols)
    abs_vals = {}
    for m in methods:
        row = df[(df["method"] == m) & (df["metric"] == sel_metric)].iloc[0]
        vals = np.array([row[c] for c in cols], dtype=float)
        abs_vals[m] = vals

    ideal = abs_vals["QFAST"]
    # 归一化 value / ideal
    norm = {m: (abs_vals[m] / ideal) for m in methods}

    # --- 绘图参数 ---
    N = len(cols)
    M = len(methods)
    group_w = 0.84
    bar_w = group_w / M
    x = np.arange(N)

    fig_w = max(8.0, 0.9 * N + 3.0)
    fig_h = 5.8
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    BAR_EDGE = "black"
    BAR_LW = 0.6

    # 绘制所有方法的柱子
    for j, m in enumerate(methods):
        heights = norm[m]
        offset = (j - (M - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w, label=m,
               color=PALETTE.get(m, None), edgecolor=BAR_EDGE, linewidth=BAR_LW, zorder=2)

    # 仅在 QFAST 柱顶标注“绝对值”
    j_ideal = methods.index("QFAST")
    offset = (j_ideal - (M - 1) / 2) * bar_w
    for i in range(N):
        val_abs = ideal[i]
        ax.annotate(_format_number(val_abs, sel_metric),
                    xy=(x[i] + offset, 1.0),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, rotation=0, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    # 坐标轴 & 图例
    # if sqrt_scale:
    #     ax.set_yscale("function", functions=(np.sqrt, lambda y: y ** 2))
    # 坐标轴 & 图例
    if sel_metric == "time_sec":
        # 自动使用 log10（更直观区分快/慢数量级差异）
        ax.set_yscale("log", base=10)
        ax.set_ylim(top=1e3, bottom=1e-6)
    else:
        ax.set_yscale("function", functions=(np.sqrt, lambda y: y ** 2))
        ax.set_ylim(bottom=0)

    ax.set_ylabel(f"{ylabel} (normalized to QFAST)")
    ax.set_xlabel("Circuit")
    ax.set_title(f"{title_suffix} (QFAST = 1.0)", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=0)
    ax.yaxis.grid(True, ls="--", lw=0.6, alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=min(5, len(handles)), frameon=False,
              fontsize=13, handlelength=2.0,
              borderpad=0.7, labelspacing=0.9, handletextpad=0.8, columnspacing=1.4)
    plt.subplots_adjust(top=0.80)

    # 输出
    out_path = Path("../results/") / f"quct_{sel_metric}_bar.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out_path}")


# Optional example usage
if __name__ == "__main__":
    plot_quct_bar(in_path="../results/synthesis_table.csv", metric="gate")
    plot_quct_bar(in_path="../results/synthesis_table.csv", metric="time")
    plot_quct_bar(in_path="../results/synthesis_table.csv", metric="depth")

