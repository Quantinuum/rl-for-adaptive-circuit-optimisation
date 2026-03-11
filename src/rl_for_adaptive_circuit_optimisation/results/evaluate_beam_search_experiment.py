import argparse

import pandas as pd

from .plots import plot_reward_result_beam_search, plot_timing_result_beam_search

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--results_file",
        type=str,
        help="Path to the results file relative to project root"
    )

    args = parser.parse_args()

    with open(args.results_file, "r") as f:
        data = pd.read_csv(f)

    reward_fig = plot_reward_result_beam_search(data)
    reward_fig.savefig("reward_results.png", bbox_inches="tight")

    timing_fig = plot_timing_result_beam_search(data)
    timing_fig.savefig("timing_results.png", bbox_inches="tight")
