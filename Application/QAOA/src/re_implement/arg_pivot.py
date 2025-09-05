

import sys
import pandas as pd


def merge_csv(file1, file2, outfile):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df2["pkid"] = df2["pkid"]+6
    # 合并：直接上下拼接
    merged = pd.concat([df1, df2], ignore_index=True)

    # 保存
    merged.to_csv(outfile, index=False)
    print(f"合并完成，已保存到：{outfile}")


import pandas as pd

def build_pivot(csv_path: str, out_path: str = "arg_pivot.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 统一列类型
    df["ARG"] = pd.to_numeric(df.get("ARG"), errors="coerce")

    # 过滤无效ARG（>10000 或为空）
    before = len(df)
    df = df[df["ARG"].notna() & (df["ARG"] <= 100000)]
    # 如需查看过滤掉多少条，可取消注释
    # print(f"Filtered {before - len(df)} rows with invalid ARG")

    # 分组计算均值
    grouped = df.groupby(["pkid", "method"], as_index=False)["ARG"].mean()

    # 透视表：行=pkid，列=method，值=ARG均值
    pivot = grouped.pivot(index="pkid", columns="method", values="ARG")

    # 映射 pkid (0..11) -> F1..K4
    pkid_map = {
        0: "F1",  1: "F2",  2: "G1",  3: "G2",  4: "K1",  5: "K2",
        6: "F3",  7: "F4",  8: "G3",  9: "G4", 10: "K3", 11: "K4",
    }
    pivot.index = pivot.index.map(lambda k: pkid_map.get(k, k))

    # 小数点后两位
    pivot = pivot.round(2)

    # 保存
    pivot.to_csv(out_path, float_format="%.2f")
    return pivot


if __name__ == "__main__":
    # merge_csv("scale_12_case_100/2_table_evaluate_second_scale.csv", "scale_34_case_100/2_table_evaluate_second_scale.csv", "final_results/table_evaluate.csv")
    pivot = build_pivot("final_results/table_evaluate.csv", "final_results/ARG.csv")
    print("已生成ARG透视表并保存到：ARG.csv")
    print(pivot)
    #

    # pivot = build_pivot("scale_34_case_100/2_table_evaluate_second_scale.csv", "scale_34_case_100/arg_scale34.csv")
    #
    # print("已生成scale3-4 arg透视表并保存到：arg_pivot.csv")
    # print(pivot)
