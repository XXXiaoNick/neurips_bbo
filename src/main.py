import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Black-box Optimization Experiment")

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/experiments.yaml",
        help="Path to the experiment YAML configuration."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run all experiments defined in the config."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze and visualize results after running."
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX table generation, produce CSV only."
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)
    sys.path.append(project_root)

    from experiment import Experiment
    from utils.table_generator import generate_final_regret_table
    from utils.visualization import generate_all_visuals

    exp = Experiment(config_path=args.config)

    if args.run:
        results = exp.run_all()
        print("Experiments completed.")
    else:
        results = exp._load_raw_results()

    if args.analyze:
        if not results:
            print("Warning: No results found. Run experiments first or check your configuration.")
            return

        exp.analyze_results(results)

        latex_flag = not args.no_latex
        generate_final_regret_table(
            results=results,
            output_dir=exp.tables_dir.as_posix(),
            file_basename="final_regret_summary",
            latex=latex_flag
        )

        generate_all_visuals(
            results=results,
            output_dir=exp.figures_dir.as_posix()
        )

        print("Analysis and visualization completed.")

if __name__ == "__main__":
    main()
