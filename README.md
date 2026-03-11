# Software Supporting Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes
Repository containing code relevant for reproducing the results of [Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes](https://arxiv.org/abs/2601.21629)

## Setup
Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Plotting scripts
Plotting scripts live under `rl_for_adaptive_circuit_optimisation.results` and read per-circuit-class
evaluation CSVs named `evaluation_results_<circuit_class>.csv`.

The evaluation CSVs are published on Zenodo:
https://zenodo.org/records/18786469

Once downloaded, experiment directories should be structured like:

```
path/to/experiments/
  experiment_1/
    evaluation_results_ordered-clifford-kak.csv
    evaluation_results_pauli.csv
    ...
  experiment_2/
    evaluation_results_ordered-clifford-kak.csv
    evaluation_results_pauli.csv
    ...
  experiment_3/
    evaluation_results_ordered-clifford-kak.csv
    evaluation_results_pauli.csv
    ...
```

Each CSV is expected to include at least:
- `total_reward`
- `best_n_2q_gates`
- baseline columns with suffixes `_reward` and `_n_2q_gates` (for example `quantinuum_default_opt_3_reward`).

### Aggregate plots across one or more experiment folders
Use `evaluate_many_experiments.py` to aggregate runs and generate plots plus summary statistics.

```bash
uv run python -m rl_for_adaptive_circuit_optimisation.results.evaluate_many_experiments \
  -b path/to/experiments \
  -f experiment_1 experiment_2 experiment_3
```

Notes:
- `-b/--base_folder` is relative to the project root.
- `-f/--folders` is a space-separated list of experiment folder names.

Outputs are written to `plots/evaluation_results_<timestamp>/` and include:
- Summary text file (`*_summary.txt`)
- PDFs of boxplots
- CSVs used to generate each plot

### Out-of-distribution (OOD) plots
Use `evaluate_ood_experiment.py` for the very large (OOD) circuits experiment.

1) Edit `DIREC_PATH` in `evaluate_ood_experiment.py` to point at the OOD results folder (relative to the
	 project root).
2) Run:

```bash
uv run python -m rl_for_adaptive_circuit_optimisation.results.evaluate_ood_experiment
```

Plots are written to `<DIREC_PATH>/plots/`.

### Beam Search Plots

Beam search plots from the paper can be reproduced by first downloading `beam_search_test_results.csv` from Zenodo as described above, then running a script such as:
```
from rl_for_adaptive_circuit_optimisation.results.evaluate_beam_search_experiment import plot_timing_result, plot_reward_result
import pandas as pd

data = pd.read_csv('/path/to/file/beam_search_test_results.csv')

fig = plot_timing_result(data)
fig = plot_reward_result(data)
```
You may also chose to run the following in the command line.
```
uv run python -m rl_for_adaptive_circuit_optimisation.results.evaluate_beam_search_experiment -r beam_search_test_results.csv
```
This will save the relevant plots in `reward_results.png` and `timing_results.png`.