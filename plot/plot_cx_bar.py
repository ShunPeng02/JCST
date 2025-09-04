# -*- coding: utf-8 -*-


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_cx_bar(in_path: str, out_path: str):
    # ----------------------- Config -----------------------

    IN_CSV = Path(in_path)
    OUT_DIR = Path(out_path)

    FIG_PATH = OUT_DIR / "cx_norm_bar.png"
    STATS_PATH = OUT_DIR / "cx_norm_stats.csv"

    METHODS = ["vf2", "sabre", "dense", "trivial"]  # 显示顺序

    FAIL_FACE = "#FDE0E0"  # FAIL 柱体的浅红填充
    FAIL_EDGE = "red"  # 斜线和边框用红色
    PALETTE = {
        "Ideal": "#B9B7B3",  # 温润灰（基线）
        "trivial": "#7E8AA2",  # 冷调雾蓝
        "sabre": "#9AA794",  # 鼠尾草绿
        "vf2": "#AB545A",  # 肉桂玫瑰
        "dense": "#A69CB0",  # 薰衣草灰
    }
    # 建议搭配：边框更深一点，增强对比
    BAR_EDGE = "#3D3D3D"
    BAR_LW = 0.7

    ERR_CAP = 3
    HATCH_ALL_FAIL = "///"

    TITLE = "Normalized CX count (Ideal = 1.0)"
    YLABEL = "Norm. Additional CX Gates"
    XLABEL = "Circuit Name"

    # 字体与布局
    BASE_FONTSIZE = 10
    TICK_FONTSIZE = 10
    LEGEND_FONTSIZE = 14
    LABEL_FONTSIZE = 11
    TITLE_FONTSIZE = 12

    # ----------------------- Load -------------------------
    df = pd.read_csv(IN_CSV)

    # 过滤出我们要的 methods
    present_methods = [m for m in METHODS if m in df["method"].unique().tolist()]
    if not present_methods:
        raise ValueError("CSV 中未发现任何预期的方法(method)。")

    # 以电路为单位获取 cx_ideal（用有效值的中位数，鲁棒）
    ideal_map = (df[df["cx_ideal"] >= 0]
                 .groupby("Circuit Name")["cx_ideal"]
                 .median())  # 绝大多数情况下每个电路一致

    # 丢弃无法归一化的电路（cx_ideal<=0 或缺失）
    valid_circuits = ideal_map[ideal_map > 0].index.tolist()
    if len(valid_circuits) < len(df["Circuit Name"].unique()):
        drop_set = set(df["Circuit Name"].unique()) - set(valid_circuits)
        print(
            f"[WARN] 跳过 {len(drop_set)} 个电路（cx_ideal<=0 或缺失）：{sorted(list(drop_set))[:5]}{' ...' if len(drop_set) > 5 else ''}")

    # 排序：按 Qubit Num 升序，再按名称
    meta = (df[df["Circuit Name"].isin(valid_circuits)]
            .groupby("Circuit Name")
            .agg(**{"Qubit Num": ("Qubit Num", "max"),
                    "Gate Num": ("Gate Num", "max")})
            .reset_index())
    circuits = (meta.sort_values(["Qubit Num", "Circuit Name"], ascending=[True, True])
                ["Circuit Name"].tolist())

    # -------------------- Aggregate stats ------------------
    # 计算 ratio = cx_constrained / cx_ideal（仅对成功样本）
    def agg_ratio(sub):
        total = len(sub)
        ideal = sub["cx_ideal"].iloc[0]
        # 成功样本：cx_added>=0 且 cx_constrained>=0 且 ideal>0
        mask_ok = (sub["cx_added"] >= 0) & (sub["cx_constrained"] >= 0) & (ideal > 0)
        ratios = sub.loc[mask_ok, "cx_constrained"] / ideal
        mean_r = ratios.mean() if len(ratios) else np.nan
        std_r = ratios.std(ddof=1) if len(ratios) > 1 else (0.0 if len(ratios) == 1 else np.nan)

        return pd.Series({
            "n_total": total,
            "n_fail": int((sub["cx_added"] == -1).sum()),
            "n_zero": int((sub["cx_added"] == 0).sum()),
            "mean_norm": mean_r,
            "std_norm": std_r,
            "Qubit Num": int(sub["Qubit Num"].max()),
            "Gate Num": int(sub["Gate Num"].max()),
            "cx_ideal": int(ideal),
        })

    stats = (df[df["Circuit Name"].isin(circuits)]
             .groupby(["Circuit Name", "method"], as_index=False)
             .apply(agg_ratio)
             .reset_index(drop=True))

    # 填齐矩阵
    all_idx = pd.MultiIndex.from_product([circuits, present_methods], names=["Circuit Name", "method"])
    stats = stats.set_index(["Circuit Name", "method"]).reindex(all_idx).reset_index()

    # 填充计数列
    for col in ["n_total", "n_fail", "n_zero", "Qubit Num", "Gate Num", "cx_ideal"]:
        stats[col] = stats[col].fillna(0).astype(int)

    # 保存统计（不含 Ideal，Ideal=1.0 可在表格里另行注明）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # stats.to_csv(STATS_PATH, index=False, encoding="utf-8")

    # ----------------------- Plot --------------------------
    N = len(circuits)
    M = 1 + len(present_methods)  # +1 for Ideal

    # 画布大小自适应
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

    # --- 先画 Ideal 基线 ---
    offset0 = (0 - (M - 1) / 2) * bar_w
    ideal_heights = np.ones(N, dtype=float)  # 全 1.0
    bars0 = ax.bar(
        x + offset0, ideal_heights, width=bar_w,
        label="Ideal (=1.0)",
        color=PALETTE["Ideal"], edgecolor=BAR_EDGE, linewidth=BAR_LW, zorder=2
    )

    # --- 再画各方法 ---
    for j, m in enumerate(present_methods, start=1):
        sub = stats[stats["method"] == m].set_index("Circuit Name").loc[circuits].reset_index()

        heights = sub["mean_norm"].to_numpy()
        yerr = sub["std_norm"].to_numpy()
        total = sub["n_total"].to_numpy()
        nfail = sub["n_fail"].to_numpy()
        nzero = sub["n_zero"].to_numpy()

        # ALL FAIL：mean_norm=NaN → 柱高 0; err 0
        mask_all_fail = (total > 0) & (nfail == total)
        heights_plot = np.where(np.isnan(heights), 0.0, heights)
        yerr_plot = np.where(np.isnan(yerr), 0.0, yerr)

        heights_plot[mask_all_fail] = 1.0

        offset = (j - (M - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, heights_plot, width=bar_w,
            label=m,
            yerr=yerr_plot, capsize=ERR_CAP,
            color=PALETTE.get(m, None), edgecolor=BAR_EDGE, linewidth=BAR_LW, zorder=2
        )

        # 失败/完美注记与 ALL FAIL 斜线
        for i, rect in enumerate(bars.patches):
            if mask_all_fail[i]:
                rect.set_facecolor(FAIL_FACE)
                rect.set_hatch("///")
                rect.set_edgecolor(FAIL_EDGE)

    # --- 轴与网格 ---
    # √ 刻度（matplotlib>=3.6 支持 function scale）
    def _sqrt(y):
        return np.sqrt(y)

    def _sq(y):
        return y ** 2

    try:
        ax.set_yscale("function", functions=(_sqrt, _sq))
    except Exception as e:
        # 回退到对数刻度（零值会不可见，可略微设定下限）
        print(f"[WARN] sqrt 刻度不可用，改用 log 刻度。原因：{e}")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)

    # ax.set_title(TITLE, pad=8)
    ax.set_ylabel(YLABEL)
    ax.set_xlabel(XLABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(circuits, rotation=45, ha="right")
    ax.yaxis.grid(True, ls="--", lw=0.6, alpha=0.6, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch  # 顶部 import 一次即可

    handles, labels = ax.get_legend_handles_labels()

    # ★ 新增 FAIL 图例项
    fail_proxy = Patch(facecolor=FAIL_FACE, edgecolor=FAIL_EDGE, hatch="///", label="FAIL")
    handles.append(fail_proxy)
    labels.append("FAIL")

    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              ncol=min(6, len(handles)), frameon=False,
              fontsize=LEGEND_FONTSIZE, handlelength=2.0,
              borderpad=0.7, labelspacing=0.9, handletextpad=0.8, columnspacing=1.6)

    # 适当降低绘图区顶部，避免图例被裁切
    plt.subplots_adjust(top=0.76 if len(handles) > 4 else 0.80)

    # --- 保存 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    plot_cx_bar("../results/b23_Tokyo20_mapping.csv", "../results")