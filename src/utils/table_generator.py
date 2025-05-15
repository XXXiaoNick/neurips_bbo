import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any


def generate_final_regret_table(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    file_basename: str = "final_regret",
    latex: bool = True
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for benchmark_name, bench_data in results.items():
        for method_name, runs_info in bench_data.items():

            final_regrets, avg_times, avg_mems = [], [], []

            for run_dict in runs_info:
                curve = run_dict.get("regret_curve", [])
                if curve:
                    final_regrets.append(curve[-1])
                if "avg_iter_time" in run_dict:
                    avg_times.append(run_dict["avg_iter_time"])
                if "avg_memory" in run_dict:
                    avg_mems.append(run_dict["avg_memory"])

            if not final_regrets:
                continue

            row = {
                "Benchmark": benchmark_name,
                "Method": method_name,
                "FinalRegret_mean": float(np.mean(final_regrets)),
                "FinalRegret_std": float(np.std(final_regrets)),
                "Time_mean": float(np.mean(avg_times)) if avg_times else np.nan,
                "Time_std": float(np.std(avg_times)) if avg_times else np.nan,
                "Memory_mean": float(np.mean(avg_mems)) if avg_mems else np.nan,
                "Memory_std": float(np.std(avg_mems)) if avg_mems else np.nan
            }

            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["Benchmark", "FinalRegret_mean"], inplace=True)

    csv_path = os.path.join(output_dir, f"{file_basename}.csv")
    df.to_csv(csv_path, index=False)

    if latex:
        latex_path = os.path.join(output_dir, f"{file_basename}.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(_df_to_latex_final_regret(df))

    return df


def _df_to_latex_final_regret(df: pd.DataFrame) -> str:

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Final Regret, Time, and Memory usage}",
        "\\label{tab:final_regret_stats}",
        "\\begin{tabular}{l l r r r r r r}",
        "\\toprule",
        "Benchmark & Method & FinalRegret$_{\\text{mean}}$ & FinalRegret$_{\\text{std}}$ & Time$_{\\text{mean}}$ & Time$_{\\text{std}}$ & Mem$_{\\text{mean}}$ & Mem$_{\\text{std}}$ \\",
        "\\midrule"
    ]

    for _, row in df.iterrows():
        fr_mean = f"{row['FinalRegret_mean']:.4f}"
        fr_std = f"{row['FinalRegret_std']:.4f}"
        tmean = f"{row['Time_mean']:.4f}" if not np.isnan(row["Time_mean"]) else "-"
        tstd = f"{row['Time_std']:.4f}" if not np.isnan(row["Time_std"]) else "-"
        mmean = f"{row['Memory_mean']:.2f}" if not np.isnan(row["Memory_mean"]) else "-"
        mstd = f"{row['Memory_std']:.2f}" if not np.isnan(row["Memory_std"]) else "-"

        lines.append(f"{row['Benchmark']} & {row['Method']} & {fr_mean} & {fr_std} & {tmean} & {tstd} & {mmean} & {mstd} \\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)