import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def plot_regret_curves(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    filename_prefix: str = "regret",
    fig_format: str = "png",
    title_prefix: str = "Pseudo-regret vs Iteration"
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for benchmark_name, bench_data in results.items():
        plt.figure(figsize=(8, 6))

        for method_name, runs in bench_data.items():
            all_curves = [r["regret_curve"] for r in runs if "regret_curve" in r and r["regret_curve"]]
            if not all_curves:
                continue

            max_len = max(len(c) for c in all_curves)
            aligned = [c + [c[-1]]*(max_len - len(c)) for c in all_curves]

            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)

            x = np.arange(len(mean_curve))
            plt.plot(x, mean_curve, label=method_name)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        plt.title(f"{title_prefix} - {benchmark_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Pseudo-regret")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{benchmark_name}.{fig_format}"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_memory_usage(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    filename_prefix: str = "memory",
    fig_format: str = "png",
    title_prefix: str = "Memory Usage"
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for benchmark_name, bench_data in results.items():
        plt.figure(figsize=(8, 6))

        for method_name, runs in bench_data.items():
            mem_curves = [r["memory_usage"] for r in runs if "memory_usage" in r and r["memory_usage"]]
            if not mem_curves:
                continue

            max_len = max(len(c) for c in mem_curves)
            aligned = [c + [c[-1]]*(max_len - len(c)) for c in mem_curves]

            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)

            x = np.arange(len(mean_curve))
            plt.plot(x, mean_curve, label=method_name)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        plt.title(f"{title_prefix} - {benchmark_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{benchmark_name}.{fig_format}"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_time_per_iteration(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    filename_prefix: str = "time_per_iter",
    fig_format: str = "png",
    title_prefix: str = "Time Per Iteration"
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for benchmark_name, bench_data in results.items():
        method_names, avg_times, std_times = [], [], []

        for method_name, runs in bench_data.items():
            times = [r["avg_iter_time"] for r in runs if "avg_iter_time" in r]
            if times:
                method_names.append(method_name)
                avg_times.append(np.mean(times))
                std_times.append(np.std(times))

        if not method_names:
            continue

        x = np.arange(len(method_names))
        plt.figure(figsize=(8, 6))
        plt.bar(x, avg_times, yerr=std_times, alpha=0.7, capsize=5)
        plt.xticks(x, method_names, rotation=30)
        plt.ylabel("Avg Time per Iter (s)")
        plt.title(f"{title_prefix} - {benchmark_name}")
        plt.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{benchmark_name}.{fig_format}"), dpi=300, bbox_inches='tight')
        plt.close()


def plot_final_regret_boxplot(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: str,
    filename_prefix: str = "final_regret_box",
    fig_format: str = "png",
    title_prefix: str = "Final Pseudo-regret Distribution"
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for benchmark_name, bench_data in results.items():
        data_for_box, labels = [], []

        for method_name, runs in bench_data.items():
            final_regrets = [r["regret_curve"][-1] for r in runs if "regret_curve" in r and r["regret_curve"]]
            if final_regrets:
                data_for_box.append(final_regrets)
                labels.append(method_name)

        if not data_for_box:
            continue

        plt.figure(figsize=(8, 6))
        plt.boxplot(data_for_box, labels=labels)
        plt.title(f"{title_prefix} - {benchmark_name}")
        plt.ylabel("Pseudo-regret")
        plt.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{benchmark_name}.{fig_format}"), dpi=300, bbox_inches='tight')
        plt.close()