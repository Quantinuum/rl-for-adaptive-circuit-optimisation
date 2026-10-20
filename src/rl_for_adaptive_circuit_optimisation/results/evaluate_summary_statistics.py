"""
This script analyzes action sequences from PPO evaluation results
and generates summary statistics including sequence frequencies and length distributions.
"""

import argparse
import csv
import glob
import logging
import os
import os.path
import re
from collections import Counter
from pathlib import Path

import git
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_project_base_directory() -> Path:
    """Get the root directory of the project."""
    repo = git.Repo(__file__, search_parent_directories=True)
    root = repo.working_tree_dir
    assert root is not None
    root_path = Path(root)
    return root_path


def read_evaluation_csv(input_filename: Path | str) -> pd.DataFrame:
    """Read an evaluation CSV file, return a DataFrame with proper parsing of action sequences.

    Args:
        input_filename: Path to the evaluation CSV file.
    """
    rows = []
    with open(input_filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        if header[-1] == "":
            header = header[:-1]

        non_action_columns = len(header) - 1
        for row in reader:
            # first columns are fixed
            fixed_cols = row[:non_action_columns]
            # join the rest back as the action string (to avoid splitting on commas inside list)
            action_str = ",".join(row[non_action_columns:])
            rows.append(fixed_cols + [action_str])

    df = pd.DataFrame(rows, columns=header)
    return df


def parse_action_list(s: str) -> list[str]:
    """
    Parse a string representation of an action list into a Python list of strings.

    The input string is expected to be in the format:
    [action1, action2, action3]
    without quotes around the action strings.

    This function:
    - Strips whitespace
    - Handles empty or NaN inputs gracefully by returning an empty list
    - Removes surrounding brackets if present
    - Splits the string on commas
    - Strips whitespace around individual actions
    - Returns a list of action strings

    Parameters
    ----------
    s: The string to parse, representing a list of actions.

    Returns
    -------
    list of str: The parsed list of action strings. Returns an empty list if input is empty or invalid.
    """
    s = s.strip()
    if not s or s.lower() == "nan" or pd.isna(s):
        return []
    # Remove brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # Split by comma, strip spaces, ignore empty strings
    return [item.strip() for item in s.split(",") if item.strip()]


def analyze_dataframe(df: pd.DataFrame, base_dir: str, base_name: str) -> None:
    """Runs your existing sequence and length analysis on a DataFrame."""

    # Convert column types
    df["episode"] = df["episode"].astype(int)
    df["total_reward"] = df["total_reward"].astype(float)
    df["length"] = df["length"].astype(int)

    # Parse action strings
    df["action"] = df["action"].apply(parse_action_list)

    # --- Sequence Frequencies ---
    sequence_counter = Counter(df["action"].apply(tuple))
    sequence_freq_df = pd.DataFrame(
        [
            {"action": list(seq), "count": count}
            for seq, count in sequence_counter.items()
        ]
    )
    sequence_freq_df.sort_values(by="count", ascending=False, inplace=True)
    sequence_freq_df.to_csv(
        os.path.join(base_dir, f"{base_name}_sequence_frequencies.csv"), index=False
    )

    # --- Length Distribution ---
    df["length"] = df["action"].apply(len)
    length_counter = Counter(df["length"])
    length_dist_df = pd.DataFrame(
        [{"length": L, "count": c} for L, c in sorted(length_counter.items())]
    )
    length_dist_df.to_csv(
        os.path.join(base_dir, f"{base_name}_length_frequencies.csv"), index=False
    )


def collect_csv_files(inputs: Path | str | list[Path | str]) -> list[str]:
    """Return a list of evaluation CSV files from one or many directories or files.

    Parameters
    ----------
    inputs: A Path, string, or list of Paths/strings representing directories or CSV files.
    Returns
    -------
    list of str: List of paths to evaluation CSV files.
    """
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    csvs = []
    for d in inputs:
        p = Path(d)
        if p.is_dir():
            pattern = str(p / "evaluation_results_*.csv")
            csvs.extend(glob.glob(pattern))
        elif p.is_file() and p.suffix == ".csv":
            csvs.append(str(p))

    # Filter out any sequence / length frequency files
    return [
        f
        for f in csvs
        if not (
            f.endswith("_sequence_frequencies.csv")
            or f.endswith("_length_frequencies.csv")
        )
    ]


def evaluate_summary_statistics(
    experiment_dir: Path | str | list[Path | str],
    circuit_class: str | None = None,
) -> None:
    """
    Analyze action sequences for all evaluation CSV files in the experiment_dir
    and generate summary statistics, if experiment_dir is a single directory.
    If experiment_dir is a list of directories:
        - Collect all evaluation_results_*.csv in all dirs
        - Merge them into one combined dataframe
        - Run sequence/length analysis once on the combined data
    This function reads the CSV file containing evaluation results, processes the action sequences,
    and generates two summary CSV files for each:
    - A CSV file with the frequency of each unique action sequence.
    - A CSV file with the frequency of action sequence lengths.
    The input CSV file is expected to have the following columns:
    - episode: The episode number.
    - total_reward: The total reward for the episode.
    - length: The length of the action sequence.
    - action: A string representation of the action sequence, formatted as a list (e.g., "[action1, action2]").

    Parameters
    ----------
    experiment_dir: Path or list of Paths to the directory containing the evaluation results.
    circuit_class: Optional string representing the circuit class for combined evaluations.

    This directory should contain CSV files named in the format "evaluation_results_*.csv".
    """
    files = collect_csv_files(experiment_dir)

    if not files:
        logger.info("No evaluation CSVs found.")
        return

    base_dir: Path | str

    if isinstance(experiment_dir, (str, Path)) and len(files) == 1:
        f = files[0]
        base_dir = os.path.dirname(f)
        base_name = os.path.splitext(os.path.basename(f))[0]
        df = read_evaluation_csv(f)
        analyze_dataframe(df, base_dir, base_name)
        return
    else:
        assert circuit_class is not None, (
            "circuit_class must be provided when analyzing multiple evaluation files"
        )
        dfs = [read_evaluation_csv(f) for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)

        folder_regex = re.compile(
            r"^(?P<datetime>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<hex>[0-9a-f]{8})_(?P<experiment_name>.*)$"
        )
        folder_names = [Path(f).parent.name for f in files]
        experiment_names = []
        for name in folder_names:
            match = folder_regex.match(name)
            if match:
                experiment_names.append(match.group("experiment_name"))

        # Deduplicate and join if multiple experiments
        experiments_str = "+".join(sorted(set(experiment_names)))

        # Folder: combined_<experiment_name>/<circuit_class>
        base_results = Path(get_project_base_directory()) / "results/ppo_training"
        base_dir = base_results / f"combined_{experiments_str}" / circuit_class
        base_dir.mkdir(parents=True, exist_ok=True)
        base_name = "combined_evaluation_results"

        analyze_dataframe(combined_df, str(base_dir), base_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="results/ppo_training/default_config",
        help="Path to results directory",
    )
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    logger.info("Evaluating...")
    evaluate_summary_statistics(experiment_dir)
