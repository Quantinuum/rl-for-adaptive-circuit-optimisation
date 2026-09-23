"""
This script evaluates multiple PPO experiment directories to generate summary statistics
about action sequences taken during evaluations. It processes all evaluation CSV files
found in the specified experiment directories and produces summary CSV files for each
circuit class, detailing the frequency of unique action sequences and their lengths.
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate_many_experiments import (
    parse_folder_list,
)
from .evaluate_summary_statistics import (
    evaluate_summary_statistics,
    get_project_base_directory,
    parse_action_list,
    read_evaluation_csv,
)
from .plots import (
    CIRCUIT_CLASS_NAME_DICT,
    DESIRED_ORDER_LIST,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_mean_pass_counts(
    grouped: dict[str, list[Path | str]], output: Path, include_do_nothing: bool
) -> None:
    """Average within each seed, then equally across seeds; SD is across seeds.

    Counts cover the complete recorded action sequence, including actions after
    the best circuit was reached. DoNothing is excluded from the chart by default.
    Existing sequence-length summaries still count all actions; both definitions
    are exported in the per-seed CSV for reproducibility.
    """

    rows = []
    for circuit_class, files in sorted(grouped.items()):
        for file in sorted(map(Path, files)):
            df = read_evaluation_csv(file)
            if df.empty:
                raise ValueError(f"No evaluation rows in {file}")
            actions = df["action"].map(parse_action_list)
            lengths = actions.map(len)
            if not np.array_equal(
                lengths.to_numpy(), df["length"].astype(int).to_numpy()
            ):
                raise ValueError(f"Recorded lengths disagree with action lists: {file}")
            passes = actions.map(
                lambda seq: sum(a.strip("\"'") != "DoNothing" for a in seq)
            )
            rows.append(
                dict(
                    circuit_class=circuit_class,
                    run=file.parent.name,
                    source_csv=str(file),
                    n_circuits=len(df),
                    mean_actions=lengths.mean(),
                    mean_passes=passes.mean(),
                )
            )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise ValueError("No evaluation CSVs found")
    column = "mean_actions" if include_do_nothing else "mean_passes"
    summary = (
        per_seed.groupby("circuit_class", sort=True)
        .agg(
            mean_count=(column, "mean"),
            sd_across_seed_means=(column, "std"),
            n_seeds=("run", "nunique"),
            n_evaluations=("n_circuits", "sum"),
        )
        .reset_index()
    )
    order = [c for c in DESIRED_ORDER_LIST if c in set(summary.circuit_class)]
    order += sorted(set(summary.circuit_class) - set(order))
    summary = summary.set_index("circuit_class").loc[order].reset_index()
    summary["includes_do_nothing"] = include_do_nothing
    output.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output.with_name(output.name + "_per_seed.csv"), index=False)
    summary.to_csv(output.with_suffix(".csv"), index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary.mean_count,
        color="#4878a8",
        width=0.65,
        yerr=summary.sd_across_seed_means.fillna(0),
        capsize=4,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_xticks(
        x,
        [CIRCUIT_CLASS_NAME_DICT.get(c, c) for c in summary.circuit_class],
        rotation=30,
        ha="center",
    )
    ax.set_xlabel("Circuit Class", fontsize=13, labelpad=7)
    ax.set_ylabel(
        "Mean Selected Actions\nPer Circuit"
        if include_do_nothing
        else "Mean Optimisation Passes\nPer Circuit",
        fontsize=13,
        labelpad=7,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(
        0,
        float((summary.mean_count + summary.sd_across_seed_means.fillna(0)).max())
        * 1.18,
    )
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mean pass counts:\n%s", summary.to_string(index=False))
    logger.info("Wrote plot and source CSVs to %s", output)


def main(args: argparse.Namespace) -> None:
    """
    Docstring for main

    Main function to evaluate multiple PPO experiment directories for summary statistics.
    It collects evaluation CSV files from the specified directories, groups them by circuit class,
    and generates summary statistics for each class.
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - hydra_experiment_name: Name of the hydra experiment (optional).
        - base_folder: Base folder for the experiments relative to project root.
        - folders: List of folders to evaluate.
    """
    base_folder = get_project_base_directory() / args.base_folder

    folders = parse_folder_list(
        hydra_experiment_name=args.hydra_experiment_name,
        base_folder=base_folder,
        experiment_folders=args.folders,
    )
    evaluation_results_regex = re.compile(
        r"^evaluation_results_(?P<circuit_class>[a-zA-Z0-9-]+)\.csv$"
    )
    grouped: dict[str, list[Path | str]] = defaultdict(list)
    for f in folders:
        for file in f.iterdir():
            match = evaluation_results_regex.match(file.name)
            if match is not None:
                circuit_class = match.group("circuit_class")
                grouped[circuit_class].append(file)

    for circuit_class, files in grouped.items():
        logger.info("Evaluation for circuit class:\n%s", circuit_class)
        evaluate_summary_statistics(files, circuit_class)

    plot_mean_pass_counts(grouped, Path(args.output), args.include_do_nothing)


if __name__ == "__main__":
    # get path to results directory using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--hydra_experiment_name",
        type=str,
        help="Name of the hydra experiment",
    )

    parser.add_argument(
        "-b",
        "--base_folder",
        type=str,
        help="Base folder for the experiments relative to project root",
        default="results/ppo_training",
    )

    parser.add_argument(
        "-f",
        "--folders",
        nargs="+",
        type=str,
        help="Space-separated list of folders to evaluate",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ppo_training/mean_passes_by_circuit_type"),
        help="Output stem for bar chart and summary CSVs",
    )
    action_count_options = parser.add_mutually_exclusive_group()
    action_count_options.add_argument(
        "--include-do-nothing",
        action="store_true",
        help="Include DoNothing in the chart (default: exclude it)",
    )
    action_count_options.add_argument(
        "--exclude-do-nothing",
        dest="include_do_nothing",
        action="store_false",
        help="Exclude DoNothing (the default)",
    )
    parser.set_defaults(include_do_nothing=False)
    args = parser.parse_args()
    assert args.base_folder is not None, "Base folder must be specified."
    main(args)
