"""This particular script is for evaluating the results of the very large
circuits (out of distribution) experiments."""

import argparse
from pathlib import Path

from rl_for_adaptive_circuit_optimisation.results.evaluate_many_experiments import (
    build_evaluation_results_dataframe,
)
from rl_for_adaptive_circuit_optimisation.results.plots import (
    plot_results_baseline_medians_only,
    plot_results_full_baseline_distribution,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--results_file",
        type=str,
        help="Path to the results file relative to project root"
    )

    args = parser.parse_args()

    DIREC_PATH = args.results_file  # Path to the folder containing the results of the 'very
    # large circuits' (OOD) experiments, relative to project root

    DIREC = Path(DIREC_PATH)
    folders = [Path(DIREC)]

    (DIREC / "plots").mkdir(parents=True, exist_ok=True)
    evaluation_results_dataframe = build_evaluation_results_dataframe(folders)

    evaluation_results_dataframe.drop(
        [c for c in evaluation_results_dataframe.columns if c.startswith("pytket")],
        axis=1,
        inplace=True,
    )

    plot_results_full_baseline_distribution(
        results_df=evaluation_results_dataframe,
        metric="reward",
        plots_folder=DIREC / "plots",
        file_prefix="ood_full_baseline_distribution",
    )

    plot_results_baseline_medians_only(
        results_df=evaluation_results_dataframe,
        metric="reward",
        plots_folder=DIREC / "plots",
        file_prefix="ood_baseline_medians_only",
    )
