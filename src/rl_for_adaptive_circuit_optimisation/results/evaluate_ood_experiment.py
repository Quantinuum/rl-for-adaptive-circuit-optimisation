"""This particular script is for evaluating the results of the very large
circuits (out of distribution) experiments."""

from pathlib import Path

from rl_for_adaptive_circuit_optimisation.results.evaluate_many_experiments import (
    build_evaluation_results_dataframe,
)
from rl_for_adaptive_circuit_optimisation.results.plots import (
    plot_results_baseline_medians_only,
    plot_results_full_baseline_distribution,
)

DIREC = Path("results/ppo_training/very_large_circuits")
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