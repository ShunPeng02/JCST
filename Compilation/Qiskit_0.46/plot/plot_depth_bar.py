
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_depth_bar(Pass: str, in_path: str, out_path: str):

    IN_CSV = Path(in_path)
    OUT_DIR = Path(out_path)
    if Pass == "routing":
        FIG_PATH = OUT_DIR / "routing_depth_norm_bar.png"
    else:
        FIG_PATH = OUT_DIR / "mapping_depth_norm_bar.png"

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
    HATCH_ALL_FAIL = "///"

    TITLE = "Normalized Depth (Ideal = 1.0)"
    YLABEL = "Norm. Depth"
    XLABEL = "Circuit Name"

    BASE_FONTSIZE = 10
    TICK_FONTSIZE = 10
    LEGEND_FONTSIZE = 14
    LABEL_FONTSIZE = 11
    TITLE_FONTSIZE = 12

    # ------------ Load ------------
    df = pd.read_csv(IN_CSV)

    present_methods = [m for m in METHODS if m in df["method"].unique().tolist()]

    # Ideal depth per circuit (median for robustness)
    ideal_map = (df[df["depth_ideal"] > 0]
                 .groupby("Circuit Name")["depth_ideal"]
                 .median())
    valid = ideal_map.index.tolist()

    # Circuit order: by qubit count then name
    meta = (df[df["Circuit Name"].isin(valid)]
            .groupby("Circuit Name")
            .agg(**{"Qubit Num": ("Qubit Num", "max")})
            .reset_index())
    circuits = (meta.sort_values(["Qubit Num", "Circuit Name"], ascending=[True, True])
                ["Circuit Name"].tolist())

    # ------------ Aggregate stats (per circuit, method) ------------
    def agg_ratio(sub):
        ideal = sub["depth_ideal"].iloc[0]
        # success: compile not failed and depth valid
        mask_ok = (sub["cx_added"] >= 0) & (sub["depth_constrained"] >= 0) & (ideal > 0)
        ratios = sub.loc[mask_ok, "depth_constrained"] / ideal
        mean_r = ratios.mean() if len(ratios) else np.nan
        std_r = ratios.std(ddof=1) if len(ratios) > 1 else (0.0 if len(ratios) == 1 else np.nan)
        # "perfect depth" equals ideal
        n_equal = int((sub.loc[mask_ok, "depth_constrained"] == ideal).sum())
        return pd.Series({
            "n_total": len(sub),
            "n_fail": int((sub["cx_added"] == -1).sum()),
            "n_equal": n_equal,
            "mean_norm": mean_r,
            "std_norm": std_r,
        })

    stats = (df[df["Circuit Name"].isin(circuits)]
             .groupby(["Circuit Name", "method"], as_index=False)
             .apply(agg_ratio).reset_index(drop=True))

    # fill matrix (circuit x method)
    all_idx = pd.MultiIndex.from_product([circuits, present_methods], names=["Circuit Name", "method"])
    stats = stats.set_index(["Circuit Name", "method"]).reindex(all_idx).reset_index()
    for c in ["n_total", "n_fail", "n_equal"]:
        stats[c] = stats[c].fillna(0).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # stats.to_csv(STATS_PATH, index=False, encoding="utf-8")

    # ------------ Plot ------------
    N = len(circuits)
    M = 1 + len(present_methods)
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
        nequal = sub["n_equal"].to_numpy()

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

    # Axes & legend
    ax.set_yscale("function", functions=(np.sqrt, lambda y: y ** 2))
    # ax.set_title(TITLE, pad=8)
    ax.set_ylabel(YLABEL)
    ax.set_xlabel(XLABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(circuits, rotation=45, ha="right")
    ax.yaxis.grid(True, ls="--", lw=0.6, alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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

    plt.subplots_adjust(top=0.78 if len(handles) > 4 else 0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out_path}")



if __name__ == "__main__":
    plot_depth_bar("mapping", "../results/b23_Tokyo20_mapping.csv", "../results")
    plot_depth_bar("routing", "../results/b23_Tokyo20_routing.csv", "../results")
