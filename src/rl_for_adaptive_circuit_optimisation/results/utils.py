"""Utility functions for project management."""

import csv
from pathlib import Path

import git
import pandas as pd


def get_project_base_directory() -> Path:
    """Get the root directory of the project."""
    repo = git.Repo(__file__, search_parent_directories=True)
    root = repo.working_tree_dir
    assert root is not None
    root_path = Path(root)
    return root_path


def read_evaluation_csv(input_filename: Path | str) -> pd.DataFrame:
    """Read an evaluation CSV file, return a DataFrame with proper parsing of
    action sequences.

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
