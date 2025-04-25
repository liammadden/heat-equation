import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

fontsize_axis = 7
fontsize_ticks = 6

default_height_to_width_ratio = (5.0**0.5 - 1.0) / 2.0

def plot_experiment(experiment, path):
    
    data_sizes = []
    model_sizes = []
    for run in experiment.runs:
        if not (run.num_samples in data_sizes):
            data_sizes.append(run.num_samples)
        if not (run.num_params in model_sizes):
            model_sizes.append(run.num_params)
    data_sizes = np.array(data_sizes) * (experiment.n_t - 1)
    data_sizes_rev = data_sizes[::-1]

    heatmap_data = []
    i = 0
    for _ in data_sizes:
        yrow = []
        for _ in model_sizes:
            training_loss = experiment.runs[i].training_losses[-1]
            yrow.append(training_loss)
            i = i + 1
        heatmap_data.append(yrow)
    heatmap_data_rev = heatmap_data[::-1]

    eps = np.max([.001, 2*np.min(np.array(heatmap_data))])
    print(eps)
    mem_caps = np.zeros(len(model_sizes))
    for i in range(len(model_sizes)):
        mem_cap = 0
        for j in range(len(data_sizes)):
            training_loss = heatmap_data[j][i]
            data_size = data_sizes[j]
            if training_loss < eps and data_size > mem_cap:
                mem_cap = data_size
        mem_caps[i] = mem_cap

    nrows = 1
    ncols = 2
    _, ax = plot_settings(nrows=nrows, ncols=ncols)

    plot_heatmap(
        axis=ax[0],
        data=np.array(heatmap_data_rev),
        xlabel="Number of parameters",
        ylabel="Number of data points",
        xticks=model_sizes,
        yticks=data_sizes_rev,
    )
    plot_lineplot(
        axis=ax[1],
        xdata=model_sizes,
        ydata=mem_caps,
        xlabel="Number of parameters",
        ylabel="Estimated memory capacity",
    )

    plt.savefig(
        os.path.join(
            path, "plots", "plot-" + experiment.model + ".pdf"
        ),
        bbox_inches="tight",
    )
    return


def plot_heatmap(axis, data, xlabel, ylabel, xticks, yticks):
    plot = axis.imshow(
        data,
        cmap="bone",
        norm=colors.LogNorm(vmin=data.min(), vmax=data.max()),
        aspect=0.75,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(xticks)), xticks)
    axis.set_yticks(range(len(yticks)), yticks)
    axis.tick_params(axis="x", rotation=90)
    cbar = plt.colorbar(plot, format="%1.3g")
    cbar.ax.tick_params()


def plot_lineplot(axis, xdata, ydata, xlabel, ylabel):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.plot(xdata, ydata)


def plot_settings(
    nrows=1, ncols=1, width=6.0, height_to_width_ratio=default_height_to_width_ratio
):
    subplot_width = width / ncols
    subplot_height = height_to_width_ratio * subplot_width
    height = subplot_height * nrows
    figsize = (width, height)

    plt.rcParams.update(
        {
            "axes.labelsize": fontsize_axis,
            "figure.figsize": figsize,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "lines.linewidth": 2,
            "lines.marker": "o",
            "xtick.labelsize": fontsize_ticks,
            "ytick.labelsize": fontsize_ticks,
            "figure.dpi": 250,
        }
    )

    return plt.subplots(nrows, ncols, constrained_layout=True)