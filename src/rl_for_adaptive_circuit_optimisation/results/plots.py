from itertools import cycle
from pathlib import Path
from typing import Literal

import matplotlib.lines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

RESULT_NAME_DICT = {
    "quantinuum_default_opt_3": "QuantinuumDefaultThree",
    "quantinuum_default_opt_2": "QuantinuumDefaultTwo",
    "full_peephole": "FullPeepholeOptimise",
    "pauli_simp": "GreedyPauliSimp",
    "kak_decomposition": "KAKDecomposition",
    "original": "Original Number of Two-Qubit Gates",
    "total": "RL Model (This Work)",
    "best": "RL Model (This Work)",
    "best_baseline": "Best Baseline",
}
CIRCUIT_CLASS_NAME_DICT = {
    "cliff-kak-squash": "Clifford-SU4-SU8",
    "clifford-kak": "Clifford-SU4",
    "iqp": "IQP",
    "ordered-clifford-kak": "Ordered-Clifford-SU4",
    "pauli": "Pauli",
    "qaoa": "QAOA",
    "random": "Random-SU4",
    "three-squash": "Random-SU8",
    "all_circuits": "Full test dataset",
}
METRIC_NAME_DICT = {
    "reward": "Cumulative Reward",
    "n_2q_gates": "Number of Two-Qubit Gates",
}
DESIRED_ORDER_LIST = [
    "random",
    "qaoa",
    "cliff-kak-squash",
    "clifford-kak",
    "ordered-clifford-kak",
    "pauli",
    "three-squash",
    "iqp",
    "all_circuits",
]


def plot_results_baseline_medians_only(
    results_df: pd.DataFrame,
    metric: Literal["reward", "n_2q_gates"],
    plots_folder: Path,
    file_prefix: str,
) -> None:
    """
    Plots the evaluation results.

    A boxplot with the metric specified on the Y axis is created. Results are plotted both for all circuits and
    for circuits grouped by class using all episodes across all experiments in the dataframe. The median
    of the baselines are plotted as highlighted points contained in the boxplot boxes.

    Also saves the results dataframe to a CSV file along with the plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results. It should have the
          following columns:
          - 'circuit_class': The class of the circuit.
          - 'total_reward': The total reward obtained during evaluation.
          - 'best_n_2q_gates': The best number of 2-qubit gates used during evaluation.
          It should also have a set of columns containing the baselines for the specified metric,
          identified by a suffix equal to the metric name.
        plots_folder (Path): Path to the folder where plots will be saved.
        file_prefix (str): Prefix for naming the output files.
    """

    principal_quantity_prefix = "total" if metric == "reward" else "best"
    principal_quantity = principal_quantity_prefix + "_" + metric

    suffix = "_" + metric

    all_circuits_results_df = results_df.copy()
    all_circuits_results_df["circuit_class"] = "all_circuits"

    _results_df = pd.concat([results_df, all_circuits_results_df], ignore_index=True)

    sns.set_theme(style="whitegrid")
    phi = 1.61803398875
    w = 9.5  # inches
    h = w / phi
    plt.figure(figsize=(w, h))
    n_classes = _results_df["circuit_class"].nunique()
    filtered_desired_order_list = [
        cls
        for cls in DESIRED_ORDER_LIST
        if cls in _results_df["circuit_class"].unique()
    ]
    width = 0.75
    ax_bp = sns.boxplot(
        x="circuit_class",
        y=principal_quantity,
        data=_results_df,
        fill=False,
        fliersize=0,
        width=width,
        order=filtered_desired_order_list,
    )

    # Enable major grid lines (y-axis only)
    ax_bp.yaxis.grid(
        which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.7
    )

    # Enable minor grid lines for more fine-grained ticks (y-axis only)
    ax_bp.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    ax_bp.yaxis.grid(
        which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5
    )

    # Add vertical grid lines at category boundaries
    for i in range(len(filtered_desired_order_list) + 1):
        ax_bp.axvline(x=i - 0.5, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    default_colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    baseline_names_with_suffix = _results_df.columns[
        _results_df.columns.str.endswith(suffix)
        & ~_results_df.columns.str.startswith(principal_quantity_prefix)
    ]
    baseline_names = baseline_names_with_suffix.str[: -len(suffix)].unique()
    baseline_colors = {k: v for k, v in zip(baseline_names_with_suffix, default_colors)}

    width_adjusted = width * 0.8
    width_advance = width_adjusted / n_classes
    for i, label in enumerate(ax_bp.get_xticklabels()):
        circuit_class_baseline_df = _results_df.loc[
            _results_df["circuit_class"] == label.get_text(), baseline_names_with_suffix
        ]
        circuit_class_median_baseline_df = circuit_class_baseline_df.median(axis=0)
        for j, (baseline, score) in enumerate(circuit_class_median_baseline_df.items()):
            if pd.isna(score):
                continue
            x = i - width_adjusted / 2 + width_advance * j
            ax_bp.scatter(
                [x],
                [score],
                color=baseline_colors[str(baseline)],
                marker="x",
                label=baseline,
                s=50,
                lw=1,
            )
            ax_bp.plot(
                [x, x],
                [_results_df[principal_quantity].min(), score],
                color=baseline_colors[str(baseline)],
                lw=1,
                alpha=0.6,
                ls="--",
            )

    plt.legend(
        [
            matplotlib.lines.Line2D(
                [],
                [],
                color=baseline_colors[b],
                marker="x",
                markersize=8,
                linewidth=0,
                markeredgewidth=1,
            )
            for b in baseline_names_with_suffix
        ],
        [RESULT_NAME_DICT[b] if b in RESULT_NAME_DICT else b for b in baseline_names],
        title="Optimisation Method (median)",
        framealpha=0.75,
        fontsize=8,
        title_fontsize=10,
        loc="upper left",
    )

    plt.xlabel("Circuit Class")
    plt.ylabel(METRIC_NAME_DICT[metric] if metric in METRIC_NAME_DICT else metric)
    plt.xticks(rotation=30)

    # Replace x-axis labels using the dictionary
    tick_locs, tick_labels = plt.xticks()
    plt.gca().set_xticks(tick_locs)  # type: ignore
    plt.gca().set_xticklabels(
        [
            (
                CIRCUIT_CLASS_NAME_DICT[x.get_text()]
                if x.get_text() in CIRCUIT_CLASS_NAME_DICT
                else x.get_text()
            )
            for x in tick_labels
        ]
    )
    plt.tight_layout()

    plt.savefig(
        plots_folder / f"{file_prefix}_{metric}_boxplots_baseline_medians_only.pdf",
        dpi=300,
    )

    with open(
        plots_folder / f"{file_prefix}_{metric}_boxplots_baseline_medians_only.csv", "w"
    ) as f:
        _results_df.to_csv(f, index=False)


def plot_results_full_baseline_distribution(
    results_df: pd.DataFrame,
    metric: Literal["reward", "n_2q_gates"],
    plots_folder: Path,
    file_prefix: str,
) -> None:
    """
    Plots the evaluation results.

    A boxplot with the metric specified on the Y axis is created. Results are plotted both for all circuits and
    for circuits grouped by class using all episodes across all experiments in the dataframe. Each baseline
    gets its own box for each circuit class and all circuits.

    Also saves the results dataframe to a CSV file along with the plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results. It should have the
          following columns:
          - 'circuit_class': The class of the circuit.
          - 'total_reward': The total reward obtained during evaluation.
          - 'best_n_2q_gates': The best number of 2-qubit gates used during evaluation.
          It should also have a set of columns containing the baselines for the specified metric,
          identified by a suffix equal to the metric name.
        plots_folder (Path): Path to the folder where plots will be saved.
        file_prefix (str): Prefix for naming the output files.
    """

    suffix = "_" + metric

    all_circuits_results_df = results_df.copy()
    all_circuits_results_df["circuit_class"] = "all_circuits"

    _results_df = pd.concat([results_df, all_circuits_results_df], ignore_index=True)

    baseline_names_with_suffix = _results_df.columns[
        _results_df.columns.str.endswith(suffix)
    ]

    _results_df = _results_df.melt(
        id_vars=[c for c in _results_df.columns if c not in baseline_names_with_suffix],
        value_vars=baseline_names_with_suffix.to_numpy(),
        var_name="baseline_method",
        value_name=metric,
    )
    _results_df["baseline_method"] = _results_df["baseline_method"].str[: -len(suffix)]

    sns.set_theme(style="whitegrid")
    phi = 1.61803398875
    w = 9.5  # inches
    h = w / phi
    plt.figure(figsize=(w, h))
    width = 0.75
    filtered_desired_order_list = [
        cls
        for cls in DESIRED_ORDER_LIST
        if cls in _results_df["circuit_class"].unique()
    ]
    ax_bp = sns.boxplot(
        x="circuit_class",
        y=metric,
        hue="baseline_method",
        data=_results_df,
        fill=False,
        fliersize=0,
        width=width,
        order=filtered_desired_order_list,
    )

    # Enable major grid lines (y-axis only)
    ax_bp.yaxis.grid(
        which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.7
    )

    # Enable minor grid lines for more fine-grained ticks (y-axis only)
    ax_bp.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    ax_bp.yaxis.grid(
        which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5
    )

    # Add vertical grid lines at category boundaries
    for i in range(len(filtered_desired_order_list) + 1):
        ax_bp.axvline(x=i - 0.5, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    handles, labels = ax_bp.get_legend_handles_labels()

    ax_bp.legend(
        handles,
        [RESULT_NAME_DICT[l] if l in RESULT_NAME_DICT else l for l in labels],
        title="Optimisation Method",
        framealpha=0.75,
        fontsize=8,
        title_fontsize=10,
        loc="upper left",
    )

    plt.xlabel("Circuit Class")
    plt.ylabel(METRIC_NAME_DICT[metric] if metric in METRIC_NAME_DICT else metric)
    plt.xticks(rotation=30)

    # Replace x-axis labels using the dictionary
    tick_locs, tick_labels = plt.xticks()
    plt.gca().set_xticks(tick_locs)  # type: ignore
    plt.gca().set_xticklabels(
        [
            (
                CIRCUIT_CLASS_NAME_DICT[x.get_text()]
                if x.get_text() in CIRCUIT_CLASS_NAME_DICT
                else x.get_text()
            )
            for x in tick_labels
        ]
    )
    plt.tight_layout()

    plt.savefig(plots_folder / f"{file_prefix}_{metric}_boxplots.pdf", dpi=300)

    with open(plots_folder / f"{file_prefix}_{metric}_boxplots.csv", "w") as f:
        _results_df.to_csv(f, index=False)


def plot_results_improvements_over_baselines(
    results_df: pd.DataFrame,
    metric: Literal["reward", "n_2q_gates"],
    plots_folder: Path,
    file_prefix: str,
) -> None:
    """
    Plots the difference between the RL model results and the baseline results.

    A boxplot with the metric specified on the Y axis is created. Results are plotted both for all circuits and
    for circuits grouped by class using all episodes across all experiments in the dataframe. Each baseline
    gets its own box for each circuit class and all circuits.

    Also saves the results dataframe to a CSV file along with the plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results. It should have the
          following columns:
          - 'circuit_class': The class of the circuit.
          - 'total_reward': The total reward obtained during evaluation.
          - 'best_n_2q_gates': The best number of 2-qubit gates used during evaluation.
          It should also have a set of columns containing the baselines for the specified metric,
          identified by a suffix equal to the metric name.
        plots_folder (Path): Path to the folder where plots will be saved.
        file_prefix (str): Prefix for naming the output files.
    """

    principal_quantity_prefix = "total" if metric == "reward" else "best"
    is_best_less = False if metric == "reward" else True

    suffix = "_" + metric
    principal_quantity = principal_quantity_prefix + suffix

    all_circuits_results_df = results_df.copy()
    all_circuits_results_df["circuit_class"] = "all_circuits"

    _results_df = pd.concat([results_df, all_circuits_results_df], ignore_index=True)

    # Get names of columns with values for the specified metric
    metric_col_names_with_suffix = _results_df.columns[
        _results_df.columns.str.endswith(suffix)
    ].to_list()

    # Get names of columns with baseline values for the specified metric
    baseline_col_names_with_suffix = [
        b for b in metric_col_names_with_suffix if b != principal_quantity
    ]

    # Compute best baseline out of all available ones
    if is_best_less:
        best_baselines = _results_df[baseline_col_names_with_suffix].min(axis=1)
    else:
        best_baselines = _results_df[baseline_col_names_with_suffix].max(axis=1)

    _results_df["best_baseline" + suffix] = best_baselines

    # Append computed best baseline to the list of metric and baseline columns
    metric_col_names_with_suffix.append("best_baseline" + suffix)
    baseline_col_names_with_suffix.append("best_baseline" + suffix)

    # Copy dataframe to compute difference between model and baselines
    _results_df_diff = _results_df.copy()

    # I am negating the subtraction in the following line because I could not find a way
    # to subtract a dataframe from a series, only the other way around
    _results_df_diff[metric_col_names_with_suffix] = -_results_df_diff[
        metric_col_names_with_suffix
    ].sub(_results_df_diff[principal_quantity], axis=0)
    _results_df_diff.drop(columns=[principal_quantity], inplace=True)

    # Reshape dataframe so that baseline values appear in a single column, differentiated by values
    # of another column "baseline_method"
    _results_df_diff = _results_df_diff.melt(
        id_vars=[
            c
            for c in _results_df_diff.columns
            if c not in baseline_col_names_with_suffix
        ],
        value_vars=baseline_col_names_with_suffix,
        var_name="baseline_method",
        value_name=metric,
    )
    # Remove suffix from baseline names in new column
    _results_df_diff["baseline_method"] = _results_df_diff["baseline_method"].str[
        : -len(suffix)
    ]

    # Box plot of distribution of differences between model and baseline for all circuits/episodes
    sns.set_theme(style="whitegrid")
    phi = 1.61803398875
    w = 9.5  # inches
    h = w / phi
    plt.figure(figsize=(w, h))
    width = 0.75
    filtered_desired_order_list = [
        cls
        for cls in DESIRED_ORDER_LIST
        if cls in _results_df["circuit_class"].unique()
    ]
    ax_bp = sns.boxplot(
        x="circuit_class",
        y=metric,
        hue="baseline_method",
        data=_results_df_diff,
        fill=False,
        fliersize=0,
        width=width,
        order=filtered_desired_order_list,
    )

    # Enable major grid lines (y-axis only)
    ax_bp.yaxis.grid(
        which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.7
    )

    # Enable minor grid lines for more fine-grained ticks (y-axis only)
    ax_bp.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    ax_bp.yaxis.grid(
        which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5
    )

    # Add vertical grid lines at category boundaries
    for i in range(len(filtered_desired_order_list) + 1):
        ax_bp.axvline(x=i - 0.5, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    handles, labels = ax_bp.get_legend_handles_labels()

    # Add legend to plot
    ax_bp.legend(
        handles,
        # Format legend labels using a dictionary defined at the top of the file
        [RESULT_NAME_DICT[l] if l in RESULT_NAME_DICT else l for l in labels],
        title="Optimisation Method",
        framealpha=0.75,
        fontsize=8,
        loc="upper left",
    )

    plt.xlabel("Circuit Class")
    plt.ylabel(
        "Difference in "
        + (METRIC_NAME_DICT[metric] if metric in METRIC_NAME_DICT else metric)
    )
    plt.xticks(rotation=30)

    # Replace x-axis labels using the dictionary
    tick_locs, tick_labels = plt.xticks()
    plt.gca().set_xticks(tick_locs)  # type: ignore
    plt.gca().set_xticklabels(
        [
            (
                CIRCUIT_CLASS_NAME_DICT[x.get_text()]
                if x.get_text() in CIRCUIT_CLASS_NAME_DICT
                else x.get_text()
            )
            for x in tick_labels
        ]
    )
    plt.tight_layout()

    # Save plot
    plt.savefig(
        plots_folder / f"{file_prefix}_{metric}_improvement_boxplots.pdf", dpi=300
    )

    # Save plot data
    with open(
        plots_folder / f"{file_prefix}_{metric}_improvement_boxplots.csv", "w"
    ) as f:
        _results_df_diff.to_csv(f, index=False)


def plot_results_relative_improvements_over_baselines(
    results_df: pd.DataFrame,
    metric: Literal["reward", "n_2q_gates"],
    plots_folder: Path,
    file_prefix: str,
) -> None:
    """
    Plots the relative difference between the RL model results and the baseline results.

    A boxplot with the metric specified on the Y axis is created. Results are plotted both for all circuits and
    for circuits grouped by class using all episodes across all experiments in the dataframe. Each baseline
    gets its own box for each circuit class and all circuits.

    Also saves the results dataframe to a CSV file along with the plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results. It should have the
          following columns:
          - 'circuit_class': The class of the circuit.
          - 'total_reward': The total reward obtained during evaluation.
          - 'best_n_2q_gates': The best number of 2-qubit gates used during evaluation.
          It should also have a set of columns containing the baselines for the specified metric,
          identified by a suffix equal to the metric name.
        plots_folder (Path): Path to the folder where plots will be saved.
        file_prefix (str): Prefix for naming the output files.
    """

    principal_quantity_prefix = "total" if metric == "reward" else "best"
    is_best_less = False if metric == "reward" else True

    suffix = "_" + metric
    principal_quantity = principal_quantity_prefix + suffix

    all_circuits_results_df = results_df.copy()
    all_circuits_results_df["circuit_class"] = "all_circuits"

    _results_df = pd.concat([results_df, all_circuits_results_df], ignore_index=True)

    # Get names of columns with values for the specified metric
    metric_col_names_with_suffix = _results_df.columns[
        _results_df.columns.str.endswith(suffix)
    ].to_list()

    # Get names of columns with baseline values for the specified metric
    baseline_col_names_with_suffix = [
        b for b in metric_col_names_with_suffix if b != principal_quantity
    ]

    # Compute best baseline out of all available ones
    if is_best_less:
        best_baselines = _results_df[baseline_col_names_with_suffix].min(axis=1)
    else:
        best_baselines = _results_df[baseline_col_names_with_suffix].max(axis=1)

    _results_df["best_baseline" + suffix] = best_baselines

    # Append computed best baseline to the list of metric and baseline columns
    metric_col_names_with_suffix.append("best_baseline" + suffix)
    baseline_col_names_with_suffix.append("best_baseline" + suffix)

    # Copy dataframe to compute difference between model and baselines
    _results_df_rel = _results_df.copy()
    baseline_originals = _results_df_rel[metric_col_names_with_suffix].copy()
    _results_df_rel[metric_col_names_with_suffix] = (
        -_results_df_rel[metric_col_names_with_suffix]
        .sub(_results_df_rel[principal_quantity], axis=0)
        .div(baseline_originals)
    )
    _results_df_rel.drop(columns=[principal_quantity], inplace=True)

    _results_df_rel = _results_df_rel.melt(
        id_vars=[
            c
            for c in _results_df_rel.columns
            if c not in baseline_col_names_with_suffix
        ],
        value_vars=baseline_col_names_with_suffix,
        var_name="baseline_method",
        value_name=metric,
    )
    _results_df_rel["baseline_method"] = _results_df_rel["baseline_method"].str[
        : -len(suffix)
    ]

    sns.set_theme(style="whitegrid")
    phi = 1.61803398875
    w = 9.5  # inches
    h = w / phi
    plt.figure(figsize=(w, h))
    width = 0.75
    filtered_desired_order_list = [
        cls
        for cls in DESIRED_ORDER_LIST
        if cls in _results_df["circuit_class"].unique()
    ]
    ax_bp = sns.boxplot(
        x="circuit_class",
        y=metric,
        hue="baseline_method",
        data=_results_df_rel,
        fliersize=0,
        fill=False,
        width=width,
        order=filtered_desired_order_list,
    )

    # Enable major grid lines (y-axis only)
    ax_bp.yaxis.grid(
        which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.7
    )

    # Enable minor grid lines for more fine-grained ticks (y-axis only)
    ax_bp.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    ax_bp.yaxis.grid(
        which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5
    )

    # Add vertical grid lines at category boundaries
    for i in range(len(filtered_desired_order_list) + 1):
        ax_bp.axvline(x=i - 0.5, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    handles, labels = ax_bp.get_legend_handles_labels()

    ax_bp.legend(
        handles,
        [RESULT_NAME_DICT[l] if l in RESULT_NAME_DICT else l for l in labels],
        title="Optimisation Method",
        framealpha=0.75,
        fontsize=8,
        loc="upper left",
    )

    plt.xlabel("Circuit Class")
    plt.ylabel(
        "Relative Difference in "
        + (METRIC_NAME_DICT[metric] if metric in METRIC_NAME_DICT else metric)
    )
    plt.xticks(rotation=30)

    # Replace x-axis labels using the dictionary
    tick_locs, tick_labels = plt.xticks()
    plt.gca().set_xticks(tick_locs)  # type: ignore
    plt.gca().set_xticklabels(
        [
            (
                CIRCUIT_CLASS_NAME_DICT[x.get_text()]
                if x.get_text() in CIRCUIT_CLASS_NAME_DICT
                else x.get_text()
            )
            for x in tick_labels
        ]
    )
    plt.tight_layout()

    # Save plot
    plt.savefig(
        plots_folder / f"{file_prefix}_{metric}_rel_improvement_boxplots.pdf", dpi=300
    )

    # Save plot data
    with open(
        plots_folder / f"{file_prefix}_{metric}_rel_improvement_boxplots.csv", "w"
    ) as f:
        _results_df_rel.to_csv(f, index=False)


def plot_reward_result_beam_search(data: pd.DataFrame) -> plt.Figure:
    """
    Plot the cumulative reward results for beam search and other optimisation methods.

    :param data: DataFrame containing the cumulative reward results.
    :return: Figure object containing the plot.
    """

    h = 4
    w = 1.3 * h

    plt.figure(figsize=(w, h))

    sns.set(rc={"figure.figsize": (w, h)})

    sns.set_theme(style="whitegrid")

    plot = sns.boxplot(
        data=data,
        y="Cumulative Reward",
        hue="Compilation Method",
        fill=False,
        showfliers=False,
        hue_order=[
            "Model",
            "QuantinuumDefaultThree",
            "QuantinuumDefaultTwo",
            "GreedySearch",
            "Depth2Width1",
            "Depth2Width2",
            "Depth3Width3",
            "Depth4Width4",
        ],
    )

    plot.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    plot.grid(which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

    # Manually create the legend
    legend_handles, _ = plot.get_legend_handles_labels()
    plt.legend(
        legend_handles,
        [
            "RL Model (This Work)",
            "QuantinuumDefaultThree",
            "QuantinuumDefaultTwo",
            "GreedySearch",
            "Depth2Width1",
            "Depth2Width2",
            "Depth3Width3",
            "Depth4Width4",
        ],
        ncol=2,
        title="Optimisation Methods",
        loc="upper left",
        bbox_to_anchor=(0, 0),
    )

    return plot.get_figure()


def plot_timing_result_beam_search(data: pd.DataFrame) -> plt.Figure:
    """Plot the timing results for beam search and other optimisation methods.

    :param data: DataFrame containing the timing results.
    :return: Figure object containing the plot.
    """

    sns.set_theme(style="whitegrid")

    plot = sns.lmplot(
        data=data,
        x="Original 2 Qubit Gate Count",
        y="Time Taken (s)",
        hue="Compilation Method",
        markers=".",
        height=4,
        aspect=1.225,
        lowess=True,
        scatter_kws={"s": 2},
        hue_order=[
            "Model",
            "QuantinuumDefaultThree",
            "QuantinuumDefaultTwo",
            "GreedySearch",
            "Depth2Width1",
            "Depth2Width2",
            "Depth3Width3",
            "Depth4Width4",
        ],
        legend=False,
    )

    legend_handles, _ = plot.axes[0][0].get_legend_handles_labels()
    lgnd = plt.legend(
        legend_handles,
        [
            "RL Model (This Work)",
            "QuantinuumDefaultThree",
            "QuantinuumDefaultTwo",
            "GreedySearch",
            "Depth2Width1",
            "Depth2Width2",
            "Depth3Width3",
            "Depth4Width4",
        ],
        ncol=2,
        title="Optimisation Methods",
        loc="upper left",
        bbox_to_anchor=(0, -0.15),
    )

    for handle in lgnd.legend_handles:
        handle.set_sizes([100])

    plot.axes[0][0].set_xlim(100, 300)
    plot.set(yscale="log")

    plot.axes[0][0].set(xlabel="Original Two-Qubit Gate Count")

    return plot.figure
