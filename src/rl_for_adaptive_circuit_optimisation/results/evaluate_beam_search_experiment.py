import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_reward_result(data: pd.DataFrame) -> plt.Figure:
    """Plot the cumulative reward results for beam search and other optimisation methods."""

    h = 4
    w = 1.3 * h

    plt.figure(figsize=(w, h))

    sns.set(rc = {'figure.figsize':(w, h)})

    sns.set_theme(style="whitegrid")

    plot = sns.boxplot(
        data=data,
        y="Cumulative Reward",
        hue="Compilation Method",
        fill=False,
        showfliers=False,
        hue_order=[
            'Model',
            'QuantinuumDefaultThree',
            'QuantinuumDefaultTwo',
            'GreedySearch',
            'Depth2Width1',
            'Depth2Width2',
            'Depth3Width3',
            'Depth4Width4',
        ]
    )

    plot.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(n=5)
    )  # 4 minor ticks between major ticks
    plot.grid(which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

    # Manually create the legend
    legend_handles, legend_labels = plot.get_legend_handles_labels()
    plt.legend(legend_handles, [
            'RL Model (This Work)',
            'QuantinuumDefaultThree',
            'QuantinuumDefaultTwo',
            'GreedySearch',
            'Depth2Width1',
            'Depth2Width2',
            'Depth3Width3',
            'Depth4Width4',
        ], ncol=2, title='Optimisation Methods', loc='upper left', bbox_to_anchor=(0, 0))

    return plot.get_figure()


def plot_timing_result(data: pd.DataFrame) -> plt.Figure:
    """Plot the timing results for beam search and other optimisation methods."""

    h = 4
    w = 1.3 * h

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
            'Model',
            'QuantinuumDefaultThree',
            'QuantinuumDefaultTwo',
            'GreedySearch',
            'Depth2Width1',
            'Depth2Width2',
            'Depth3Width3',
            'Depth4Width4',
        ],
        legend=False
    )

    legend_handles, _ = plot.axes[0][0].get_legend_handles_labels()
    lgnd = plt.legend(
        legend_handles, [
            'RL Model (This Work)',
            'QuantinuumDefaultThree',
            'QuantinuumDefaultTwo',
            'GreedySearch',
            'Depth2Width1',
            'Depth2Width2',
            'Depth3Width3',
            'Depth4Width4',
        ], ncol=2, title='Optimisation Methods', loc='upper left', bbox_to_anchor=(0, -0.15)
    )

    for handle in lgnd.legend_handles:
        handle.set_sizes([100])

    plot.axes[0][0].set_xlim(100, 300)
    plot.set(yscale="log")

    plot.axes[0][0].set(xlabel='Original Two-Qubit Gate Count')

    return plot.figure