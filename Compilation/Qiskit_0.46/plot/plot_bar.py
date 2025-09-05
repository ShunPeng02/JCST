# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_norm_bar(metric: str, Pass: str, in_path: str, out_path: str):
    """
    metric: "cx" or "depth"
    Pass:   "routing" or "mapping"（决定输出文件名前缀）
    """
    assert metric in ("cx", "depth"), "metric 只能是 'cx' 或 'depth'"

    # ----------------------- Config -----------------------
    IN_CSV = Path(in_path)
    OUT_DIR = Path(out_path)

    # 输出文件名与标题/坐标轴
    if metric == "cx":
        FIG_PATH = OUT_DIR / (f"{Pass}_cx_norm_bar.png" if Pass == "routing" else "mapping_cx_norm_bar.png")
        TITLE = "Normalized CX count (Ideal = 1.0)"
        YLABEL = "Norm. Additional CX Gates"
    else:  # depth
        FIG_PATH = OUT_DIR / (f"{Pass}_depth_norm_bar.png" if Pass == "routing" else "mapping_depth_norm_bar.png")
        TITLE = "Normalized Depth (Ideal = 1.0)"
        YLABEL = "Norm. Depth"

    XLABEL = "Circuit Name"

    METHODS = ["vf2", 'basic', "sabre", "dense", "trivial", 'lookahead', 'stochastic']  # 显示顺序

    FAIL_FACE = "#FDE0E0"  # FAIL 柱体的浅红填充
    FAIL_EDGE = "red"  # 斜线和边框用红色
    PALETTE = {
        "Ideal": "#CFCFCF",  # 温润灰（基线）
        "trivial": "#87CEFA",  # 浅蓝
        "sabre": "#4682B4",  # 深蓝
        "vf2": "#FF0000",  # 红色
        "dense": "#9BCD9B",  # 绿色
        "basic": "#FF0000",  # 红色
        "lookahead": "#9BCD9B",  # 深绿
        "stochastic": "#87CEFA",  # 浅蓝
    }
    BAR_EDGE = "#3D3D3D"
    BAR_LW = 0.7
    ERR_CAP = 3

    BASE_FONTSIZE = 10
    TICK_FONTSIZE = 10
    LEGEND_FONTSIZE = 14
    LABEL_FONTSIZE = 11
    TITLE_FONTSIZE = 12

    # ----------------------- Load -------------------------
    df = pd.read_csv(IN_CSV)

    present_methods = [m for m in METHODS if m in df["method"].unique().tolist()]
    if not present_methods:
        raise ValueError("CSV 中未发现任何预期的方法(method)。")

    # 字段名按 metric 切换
    ideal_col = f"{metric}_ideal"
    constr_col = f"{metric}_constrained"

    # 以电路为单位获取 ideal（用有效值中位数，鲁棒）
    # cx: ideal>0；depth: ideal>0
    ideal_map = (df[df[ideal_col] > 0]
                 .groupby("Circuit Name")[ideal_col]
                 .median())

    valid_circuits = ideal_map.index.tolist()
    # Circuit 顺序：Qubit Num 升序，其次名字
    meta = (df[df["Circuit Name"].isin(valid_circuits)]
            .groupby("Circuit Name")
            .agg(**{"Qubit Num": ("Qubit Num", "max"),
                    "Gate Num": ("Gate Num", "max")})
            .reset_index())
    circuits = (meta.sort_values(["Qubit Num", "Circuit Name"], ascending=[True, True])
                ["Circuit Name"].tolist())

    # -------------------- Aggregate stats ------------------
    # 统一的 ratio= constrained / ideal（仅成功样本）
    def agg_ratio(sub):
        total = len(sub)
        ideal = sub[ideal_col].iloc[0]
        # 成功样本：cx_added>=0 且 constrained>=0 且 ideal>0
        mask_ok = (sub["cx_added"] >= 0) & (sub[constr_col] >= 0) & (ideal > 0)
        ratios = sub.loc[mask_ok, constr_col] / ideal
        mean_r = ratios.mean() if len(ratios) else np.nan
        std_r = ratios.std(ddof=1) if len(ratios) > 1 else (0.0 if len(ratios) == 1 else np.nan)

        # 仅用于补充信息（保持两个旧脚本的口径）
        extra = {}
        if metric == "cx":
            extra["n_zero"] = int((sub["cx_added"] == 0).sum())
        else:  # depth
            extra["n_equal"] = int((sub.loc[mask_ok, constr_col] == ideal).sum())

        return pd.Series({
            "n_total": total,
            "n_fail": int((sub["cx_added"] == -1).sum()),
            "mean_norm": mean_r,
            "std_norm": std_r,
            **extra
        })

    stats = (df[df["Circuit Name"].isin(circuits)]
             .groupby(["Circuit Name", "method"], as_index=False)
             .apply(agg_ratio)
             .reset_index(drop=True))

    # 填齐矩阵
    all_idx = pd.MultiIndex.from_product([circuits, present_methods], names=["Circuit Name", "method"])
    stats = stats.set_index(["Circuit Name", "method"]).reindex(all_idx).reset_index()

    # 填充计数列
    fill_cols = ["n_total", "n_fail"]
    if metric == "cx":
        fill_cols.append("n_zero")
    else:
        fill_cols.append("n_equal")
    for c in fill_cols:
        stats[c] = stats[c].fillna(0).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------- Plot --------------------------
    N = len(circuits)
    M = 1 + len(present_methods)  # +1 for Ideal

    fig_w = max(10.0, 0.55 * N + 3.0)
    fig_h = 6.8 if N <= 22 else 8.5 if N <= 35 else 10.0

    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(N)
    group_w = 0.84
    bar_w = group_w / M

    # Ideal (=1.0)
    offset0 = (0 - (M - 1) / 2) * bar_w
    ax.bar(x + offset0, np.ones(N), width=bar_w, label="Ideal (=1.0)",
           color=PALETTE["Ideal"], edgecolor=BAR_EDGE, linewidth=BAR_LW, zorder=2)

    # Methods
    for j, m in enumerate(present_methods, start=1):
        sub = stats[stats["method"] == m].set_index("Circuit Name").loc[circuits].reset_index()

        heights = sub["mean_norm"].to_numpy()
        yerr = sub["std_norm"].to_numpy()
        total = sub["n_total"].to_numpy()
        nfail = sub["n_fail"].to_numpy()

        # ALL FAIL：全部失败柱显示为 1.0，且用斜线+红框
        mask_all_fail = (total > 0) & (nfail == total)
        heights_plot = np.where(np.isnan(heights), 0.0, heights)
        yerr_plot = np.where(np.isnan(yerr), 0.0, yerr)

        heights_plot[mask_all_fail] = 1.0

        offset = (j - (M - 1) / 2) * bar_w
        bars = ax.bar(x + offset, heights_plot, width=bar_w, label=m,
                      yerr=yerr_plot, capsize=ERR_CAP,
                      color=PALETTE.get(m, None), edgecolor=BAR_EDGE,
                      linewidth=BAR_LW, zorder=2)

        for i, rect in enumerate(bars.patches):
            if mask_all_fail[i]:
                rect.set_facecolor(FAIL_FACE)
                rect.set_hatch("///")
                rect.set_edgecolor(FAIL_EDGE)

    # y 轴开平方刻度（兼容性处理）
    def _sqrt(y):
        return np.sqrt(y)

    def _sq(y):
        return y ** 2

    try:
        ax.set_yscale("function", functions=(_sqrt, _sq))
    except Exception as e:
        print(f"[WARN] sqrt 刻度不可用，改用 log 刻度。原因：{e}")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)

    # ax.set_title(TITLE, pad=8)  # 原脚本注释了标题
    ax.set_ylabel(YLABEL)
    ax.set_xlabel(XLABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(circuits, rotation=45, ha="right")
    ax.yaxis.grid(True, ls="--", lw=0.6, alpha=0.6, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 图例（含 FAIL 样例）
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=FAIL_FACE, edgecolor=FAIL_EDGE, hatch="///", label="FAIL"))
    labels.append("FAIL")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              ncol=min(6, len(handles)), frameon=False,
              fontsize=LEGEND_FONTSIZE, handlelength=2.0,
              borderpad=0.7, labelspacing=0.9, handletextpad=0.8, columnspacing=1.6)

    plt.subplots_adjust(top=0.76 if len(handles) > 4 else 0.80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"[Saved] {FIG_PATH}")


if __name__ == "__main__":
    # 与原脚本调用保持一致性示例（你可按需修改路径）
    # depth
    plot_norm_bar("depth", "mapping", "../results/b23_Tokyo20_mapping.csv", "../results")
    plot_norm_bar("depth", "routing", "../results/b23_Tokyo20_routing.csv", "../results")
    # cx
    plot_norm_bar("cx", "mapping", "../results/b23_Tokyo20_mapping.csv", "../results")
    plot_norm_bar("cx", "routing", "../results/b23_Tokyo20_routing.csv", "../results")
