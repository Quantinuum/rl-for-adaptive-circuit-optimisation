"""
This script generates comparative plots.
It reads evaluation results from specified experiment directories, aggregates
the data,
and creates boxplots for total rewards and best 2-q gate counts across
different circuit classes, comparing against baseline methods in the former
case.

You must provide a list of specific folders via the -f or --folders flag.


If only one folder is passed via -f / --folders, it generates the plots using
only that experiment's data.
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from rl_for_adaptive_circuit_optimisation.results.plots import (
    plot_results_baseline_medians_only,
    plot_results_full_baseline_distribution,
    plot_results_improvements_over_baselines,
    plot_results_relative_improvements_over_baselines,
)
from rl_for_adaptive_circuit_optimisation.results.utils import (
    get_project_base_directory,
    read_evaluation_csv,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

for lib_logger_name in ["pytket-benchmarking", "torch", "gymnasium"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)

CIRCUIT_CLASSES_BY_DATASET: dict[str, list[str]] = {
    "iqp_qaoa_rand_pauli_long_ordered_kak_cliff_squash_small": [
        "cliff-kak-squash",
        "clifford-kak",
        "iqp",
        "ordered-clifford-kak",
        "pauli",
        "qaoa",
        "random",
        "three-squash",
    ],
    "iqp_qaoa_rand_pauli_clifford_squash_small": [
        "clifford-kak",
        "iqp",
        "pauli",
        "qaoa",
        "random",
        "three-squash",
    ],
}


def parse_folder_list(
    base_folder: Path,
    experiment_folders: list[str],
) -> list[Path]:
    """Parse the list of experiment folders to evaluate.

    Those specific folders with the base folder prepended
    are returned if they exist, otherwise an exception is raised.

    Args:
        base_folder: Base folder where experiments are located.
        experiment_folders: List of specific folder names to evaluate.
    """
    folders = []
    for folder in experiment_folders:
        folder_path = base_folder / folder
        if folder_path.is_dir():
            folders.append(folder_path)
        else:
            raise ValueError(f"Folder does not exist: {folder_path}")
    return folders

def compute_evaluation_statistics(all_results_df: pd.DataFrame) -> str:
    """Compute and return evaluation statistics and baselines across all experiments.

    Computes mean, standard deviation, median, and quantiles for total rewards,
    best 2-q gate counts, and baselines, grouped by circuit class and for all circuit classes.

    Also computes the number and percentage of experiments where RL improved over baselines
    and over all circuits.

    Args:
        all_results_df: DataFrame containing evaluation results from all experiments.
    """

    grouped_results_circuit_class = all_results_df.groupby("circuit_class")
    # Suffixes of data frame columns identifying each metric of interest
    suffixes = ["_reward", "_n_2q_gates"]
    # Data frame columns identifying RL results for each metric of interest
    rl_result_columns = ["total_reward", "best_n_2q_gates"]
    # Sign indicating whether lower is better for each metric of interest
    is_best_less = [False, True]
    _is_best_less_sign = [1 - 2 * x for x in is_best_less]

    summary = ""

    def _generate_summary_for_circuit_class(
        df: pd.DataFrame, column: str, rl_result_column: str, sign: int
    ) -> str:
        """Helper function to append summary statistics for a given column.

        Args:
            df: DataFrame containing evaluation results.
            column: Column name to compute statistics for.
            rl_result_column: Column name for RL results to compare against.
            sign: Sign indicating whether lower is better for the metric.
        Returns:
            A string containing the summary statistics for the column.
        """
        to_append = ""
        mean = df[column].mean()
        std_error = cast(float, df[column].sem())
        std_deviation = df[column].std()
        median = df[column].median()
        to_append += f"Metric: {column}\n"
        to_append += f"{column}: Mean {mean} ± standard error {std_error}, standard deviation {std_deviation}\n"
        to_append += f"{column}: Median {median}, 25th quantile {df[column].quantile(0.25)}, 75th quantile {df[column].quantile(0.75)}\n"
        if not column == rl_result_column:
            n_circuits_improved_over_baseline = (
                sign * (df[rl_result_column] - df[column]) > 0
            ).sum()
            to_append += f"{column}: RL improved over baseline: {n_circuits_improved_over_baseline} out of {len(df)}\n"
            to_append += f"{column}: RL percentage improved over baseline: {100 * n_circuits_improved_over_baseline / len(df)}%\n"
        to_append += "\n"
        return to_append

    # Compute summary statistics grouped by circuit class
    for k, df in grouped_results_circuit_class:
        summary += f"Circuit class: {k}\n"
        for s, rl_c, sign in zip(suffixes, rl_result_columns, _is_best_less_sign):
            for c in all_results_df.columns:
                if not c.endswith(s):
                    continue
                summary += _generate_summary_for_circuit_class(df, c, rl_c, sign)
            summary += "\n"
        summary += "\n"

    # Compute summary statistics for all circuits
    summary += "\nAll circuits:\n"
    for s, rl_c, less_is_best, sign in zip(
        suffixes, rl_result_columns, is_best_less, _is_best_less_sign
    ):
        for c in all_results_df.columns:
            if not c.endswith(s):
                continue
            summary += _generate_summary_for_circuit_class(
                all_results_df, c, rl_c, sign
            )
        all_baseline_result_columns = all_results_df[
            all_results_df.columns[
                (all_results_df.columns.str.endswith(s))
                & (all_results_df.columns != rl_c)
            ]
        ]

        # Take best out of all baselines
        if less_is_best:
            best_baseline = all_baseline_result_columns.min(axis=1)
        else:
            best_baseline = all_baseline_result_columns.max(axis=1)

        # Compute how many circuits RL improved over best compilation baseline
        summary += "\n"
        summary += f"Number of circuits that model improved over ANY baseline: {(sign *(all_results_df[rl_c] - best_baseline) > 0).sum()} out of {len(all_results_df)}\n"
        summary += f"Percentage of circuits improved by the model over ANY baseline: {100 * (sign *(all_results_df[rl_c] - best_baseline) > 0).sum() / len(all_results_df)}%\n"
        summary += "\n"
    summary += "\n"

    return summary


def build_evaluation_results_dataframe(folders: list[Path]) -> pd.DataFrame:
    """Build a combined evaluation results DataFrame from multiple experiment folders.

    Args:
        folders: List of experiment folder paths to read evaluation results from.
    """
    evaluation_results_regex = re.compile(
        r"^evaluation_results_(?P<circuit_class>[a-zA-Z0-9-]+)\.csv$"
    )
    all_results = []
    for i, f in enumerate(folders):
        for file in f.iterdir():
            match = evaluation_results_regex.match(file.name)
            if match is not None:
                circuit_class = match.group("circuit_class")
                logger.info(
                    f"Loaded evaluation results for circuit class {circuit_class} in folder {f}:"
                )
                results = read_evaluation_csv(file)
                for c in results.columns:
                    if c.endswith("_reward"):
                        results[c] = results[c].astype(float)
                    if c.endswith("_n_2q_gates"):
                        results[c] = results[c].astype(int)
                results["experiment_folder"] = f.name
                results["circuit_class"] = circuit_class
                results["run_index"] = i
                all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df


if __name__ == "__main__":
    # get path to results directory using argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--base_folder",
        type=str,
        help="Base folder for the experiments relative to project root"
    )

    parser.add_argument(
        "-f",
        "--folders",
        nargs="+",
        type=str,
        help="Space-separated list of folders to evaluate"
    )

    args = parser.parse_args()

    assert args.base_folder is not None, "Base folder must be specified."
    base_folder = get_project_base_directory() / args.base_folder

    folders = parse_folder_list(
        base_folder=base_folder,
        experiment_folders=args.folders,
    )

    all_results_df = build_evaluation_results_dataframe(folders)

    all_results_df.drop(
        [c for c in all_results_df.columns if c.startswith("pytket")],
        axis=1,
        inplace=True,
    )

    baseline_suffix = "_n_2q_gates"
    baseline_names = [
        col[: -len(baseline_suffix)]
        for col in all_results_df.columns
        if col.endswith(baseline_suffix)
    ]

    # Get current date and time to name the output plots
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary = compute_evaluation_statistics(all_results_df)

    logger.info("Evaluation Summary:\n%s", summary)

    file_prefix = f"evaluation_results_{time}"
    plots_folder = get_project_base_directory() / "plots" / file_prefix
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Save summary to a text file
    summary_file = plots_folder / f"{file_prefix}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)

    # Plots showing improvement of RL model over baselines on a per-circuit basis
    plot_results_improvements_over_baselines(
        all_results_df, "reward", plots_folder=plots_folder, file_prefix=file_prefix
    )
    plot_results_improvements_over_baselines(
        all_results_df, "n_2q_gates", plots_folder=plots_folder, file_prefix=file_prefix
    )

    # Plots showing relative improvement of RL model over baselines on a per-circuit basis
    plot_results_relative_improvements_over_baselines(
        all_results_df, "reward", plots_folder=plots_folder, file_prefix=file_prefix
    )
    plot_results_relative_improvements_over_baselines(
        all_results_df, "n_2q_gates", plots_folder=plots_folder, file_prefix=file_prefix
    )

    # Plots where baseline distribution is included as boxes in a boxplot
    plot_results_full_baseline_distribution(
        all_results_df, "reward", plots_folder=plots_folder, file_prefix=file_prefix
    )
    plot_results_full_baseline_distribution(
        all_results_df, "n_2q_gates", plots_folder=plots_folder, file_prefix=file_prefix
    )

    # Plots where median of baselines is shown as points in the boxplot
    plot_results_baseline_medians_only(
        all_results_df, "reward", plots_folder=plots_folder, file_prefix=file_prefix
    )
    plot_results_baseline_medians_only(
        all_results_df, "n_2q_gates", plots_folder=plots_folder, file_prefix=file_prefix
    )
