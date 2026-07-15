"""
Tools to visualize the results for the sample size of a MLMC simulation.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

MARKER = ["o", "x", "s", "d", "*"]


def create_figures_N_l_error(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2,
        jump_rates, identifier):
    """
    This function creates a figure that displays the N_l-level
    dependency for barrier and asian options for different jump rates and epsilon=0.05.
    Args:
        results_barrier_1: This list contains dfs (one for each jump rate)
        that store information about N_l for a barrier option for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump rate)
        that store information about N_l for a barrier option for the second scheme.
        results_asian_1: This list contains dfs (one for each jump rate)
        that store information about N_l for an asian option for the first scheme.
        results_asian_2: This list contains dfs (one for each jump rate)
        that store information about N_l for an asian option for the second scheme.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    max_l = 0

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier 1
        levels_barrier_1 = results_barrier_1[i].index.values
        if len(levels_barrier_1) > max_l:
            max_l = len(levels_barrier_1)
        sample_size_barrier_1 = results_barrier_1[i]["N_l"].to_numpy()
        axs[0, 0].plot(levels_barrier_1, sample_size_barrier_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier 2
        levels_barrier_2 = results_barrier_2[i].index.values
        if len(levels_barrier_2) > max_l:
            max_l = len(levels_barrier_2)
        sample_size_barrier_2 = results_barrier_2[i]["N_l"].to_numpy()
        axs[1, 0].plot(levels_barrier_2, sample_size_barrier_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 1
        levels_asian_1 = results_asian_1[i].index.values
        if len(levels_asian_1) > max_l:
            max_l = len(levels_asian_1)
        sample_size_asian_1 = results_asian_1[i]["N_l"].to_numpy()
        axs[0, 1].plot(levels_asian_1, sample_size_asian_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 2
        levels_asian_2 = results_asian_2[i].index.values
        if len(levels_asian_2) > max_l:
            max_l = len(levels_asian_2)
        sample_size_asian_2= results_asian_2[i]["N_l"].to_numpy()
        axs[1, 1].plot(levels_asian_2, sample_size_asian_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_xlabel(r"$l$")
    axs[0, 0].set_ylabel(r"$N_l$")

    axs[1, 0].set_xlabel(r"$l$")
    axs[1, 0].set_ylabel(r"$N_l$")

    axs[0, 1].set_xlabel(r"$l$")
    axs[0, 1].set_ylabel(r"$N_l$")

    axs[1, 1].set_xlabel(r"$l$")
    axs[1, 1].set_ylabel(r"$N_l$")

    if identifier == 0:
        axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")

    elif identifier == 1:
        axs[0, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")

    else:
        axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")


    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, max_l)
        ax.set_xticks(np.arange(1, max_l+1, 2))
        ax.set_yscale("log")

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if identifier == 0:
        plt.savefig("Plots_Num_Samples/samples_fixed_error_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Num_Samples/samples_fixed_error_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Num_Samples/samples_fixed_error_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_N_l_jump_rate(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2,
        error_bounds, identifier):
    """
    This function creates a figure that displays the N_l-level
    dependency for barrier and asian options for different error bounds and lambda=10.
    Args:
        results_barrier_1: This list contains dfs (one for each error bound)
        that store information about N_l for a barrier option for the first scheme.
        results_barrier_2: This list contains dfs (one for each error bound)
        that store information about N_l for a barrier option for the second scheme.
        results_asian_1: This list contains dfs (one for each error bound)
        that store information about N_l for an asian option for the first scheme.
        results_asian_2: This list contains dfs (one for each error bound)
        that store information about N_l for an asian option for the second scheme.
        error_bounds: This list contains different values for the error bounds.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    max_l = 0

    for i, bound in enumerate(error_bounds):
        marker = MARKER[i]

        # Barrier 1
        levels_barrier_1 = results_barrier_1[i].index.values
        if len(levels_barrier_1) > max_l:
            max_l = len(levels_barrier_1)
        sample_size_barrier_1 = results_barrier_1[i]["N_l"].to_numpy()
        axs[0, 0].plot(levels_barrier_1, sample_size_barrier_1,
                       label=fr"$\epsilon = {bound}$", marker=marker)

        # Barrier 2
        levels_barrier_2 = results_barrier_2[i].index.values
        if len(levels_barrier_2) > max_l:
            max_l = len(levels_barrier_2)
        sample_size_barrier_2 = results_barrier_2[i]["N_l"].to_numpy()
        axs[1, 0].plot(levels_barrier_2, sample_size_barrier_2,
                       label=fr"$\epsilon = {bound}$", marker=marker)

        # Asian 1
        levels_asian_1 = results_asian_1[i].index.values
        if len(levels_asian_1) > max_l:
            max_l = len(levels_asian_1)
        sample_size_asian_1 = results_asian_1[i]["N_l"].to_numpy()
        axs[0, 1].plot(levels_asian_1, sample_size_asian_1,
                       label=fr"$\epsilon = {bound}$", marker=marker)

        # Asian 2
        levels_asian_2 = results_asian_2[i].index.values
        if len(levels_asian_2) > max_l:
            max_l = len(levels_asian_2)
        sample_size_asian_2 = results_asian_2[i]["N_l"].to_numpy()
        axs[1, 1].plot(levels_asian_2, sample_size_asian_2,
                       label=fr"$\epsilon = {bound}$", marker=marker)

    axs[0, 0].set_xlabel(r"$l$")
    axs[0, 0].set_ylabel(r"$N_l$")

    axs[1, 0].set_xlabel(r"$l$")
    axs[1, 0].set_ylabel(r"$N_l$")

    axs[0, 1].set_xlabel(r"$l$")
    axs[0, 1].set_ylabel(r"$N_l$")

    axs[1, 1].set_xlabel(r"$l$")
    axs[1, 1].set_ylabel(r"$N_l$")

    if identifier == 0:
        axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")

    elif identifier == 1:
        axs[0, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")

    else:
        axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
        axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
        axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, max_l)
        ax.set_xticks(np.arange(1, max_l+1, 2))
        ax.set_yscale("log")

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if identifier == 0:
        plt.savefig("Plots_Num_Samples/samples_fixed_jump_rate_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Num_Samples/samples_fixed_jump_rate_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Num_Samples/samples_fixed_jump_rate_c_f.pdf", bbox_inches="tight")

    plt.show()


def plots_main():
    """
    Main function to create and save relevant plots.
    """
    jump_rates = [1, 5, 10, 15, 20]
    error_bounds = [0.1, 0.075, 0.05, 0.025]

    # Load results
    results_barrier_fixed_grid = []
    results_barrier_jump_adapted = []
    results_barrier_cutoff = []

    results_asian_fixed_grid = []
    results_asian_jump_adapted = []
    results_asian_cutoff = []

    for bound in error_bounds:
        results_barrier_fixed_grid_temp = []
        results_barrier_jump_adapted_temp = []
        results_barrier_cutoff_temp = []

        results_asian_fixed_grid_temp = []
        results_asian_jump_adapted_temp = []
        results_asian_cutoff_temp = []

        for jump_rate in jump_rates:
            results_df_barrier_fixed = pd.read_feather(
                f"MLMC_Results/Results/results_barrier_fixed_{bound}_{jump_rate}.feather")
            results_barrier_fixed_grid_temp.append(
                results_df_barrier_fixed)

            results_df_barrier_adapted = pd.read_feather(
                f"MLMC_Results/Results/results_barrier_adapted_{bound}_{jump_rate}.feather")
            results_barrier_jump_adapted_temp.append(
                results_df_barrier_adapted)

            results_df_barrier_cutoff = pd.read_feather(
                f"MLMC_Results/Results/results_barrier_cutoff_{bound}_{jump_rate}.feather")
            results_barrier_cutoff_temp.append(
                results_df_barrier_cutoff)

            results_df_asian_fixed = pd.read_feather(
                f"MLMC_Results/Results/results_asian_fixed_{bound}_{jump_rate}.feather")
            results_asian_fixed_grid_temp.append(
                results_df_asian_fixed)

            results_df_asian_adapted = pd.read_feather(
                f"MLMC_Results/Results/results_asian_adapted_{bound}_{jump_rate}.feather")
            results_asian_jump_adapted_temp.append(
                results_df_asian_adapted)

            results_df_asian_cutoff = pd.read_feather(
                f"MLMC_Results/Results/results_asian_cutoff_{bound}_{jump_rate}.feather")
            results_asian_cutoff_temp.append(
                results_df_asian_cutoff)

        results_barrier_fixed_grid.append(results_barrier_fixed_grid_temp)
        results_barrier_jump_adapted.append(results_barrier_jump_adapted_temp)
        results_barrier_cutoff.append(results_barrier_cutoff_temp)

        results_asian_fixed_grid.append(results_asian_fixed_grid_temp)
        results_asian_jump_adapted.append(results_asian_jump_adapted_temp)
        results_asian_cutoff.append(results_asian_cutoff_temp)

    args_error_nc = {"results_barrier_1": results_barrier_fixed_grid[2],
                     "results_barrier_2": results_barrier_jump_adapted[2],
                     "results_asian_1": results_asian_fixed_grid[2],
                     "results_asian_2": results_asian_jump_adapted[2],
                     "jump_rates": jump_rates,
                     "identifier": 0}

    create_figures_N_l_error(**args_error_nc)

    args_error_c_a = {"results_barrier_1": results_barrier_jump_adapted[2],
                      "results_barrier_2": results_barrier_cutoff[2],
                      "results_asian_1": results_asian_jump_adapted[2],
                      "results_asian_2": results_asian_cutoff[2],
                      "jump_rates": jump_rates,
                      "identifier": 1}

    create_figures_N_l_error(**args_error_c_a)

    args_error_c_f = {"results_barrier_1": results_barrier_fixed_grid[2],
                      "results_barrier_2": results_barrier_cutoff[2],
                      "results_asian_1": results_asian_fixed_grid[2],
                      "results_asian_2": results_asian_cutoff[2],
                      "jump_rates": jump_rates,
                      "identifier": 2}

    create_figures_N_l_error(**args_error_c_f)

    results_barrier_fixed_grid_10 = [results_barrier_fixed_grid[i][2] for i in range(len(error_bounds))]
    results_barrier_jump_adapted_10 = [results_barrier_jump_adapted[i][2] for i in range(len(error_bounds))]
    results_barrier_cutoff_10 = [results_barrier_cutoff[i][2] for i in range(len(error_bounds))]

    results_asian_fixed_grid_10 = [results_asian_fixed_grid[i][2] for i in range(len(error_bounds))]
    results_asian_jump_adapted_10 = [results_asian_jump_adapted[i][2] for i in range(len(error_bounds))]
    results_asian_cutoff_10 = [results_asian_cutoff[i][2] for i in range(len(error_bounds))]

    args_jump_rate_nc = {"results_barrier_1": results_barrier_fixed_grid_10,
                         "results_barrier_2": results_barrier_jump_adapted_10,
                         "results_asian_1": results_asian_fixed_grid_10,
                         "results_asian_2": results_asian_jump_adapted_10,
                         "error_bounds": error_bounds,
                         "identifier": 0}

    create_figures_N_l_jump_rate(**args_jump_rate_nc)

    args_jump_rate_c_a = {"results_barrier_1": results_barrier_jump_adapted_10,
                          "results_barrier_2": results_barrier_cutoff_10,
                          "results_asian_1": results_asian_jump_adapted_10,
                          "results_asian_2": results_asian_cutoff_10,
                          "error_bounds": error_bounds,
                          "identifier": 1}

    create_figures_N_l_jump_rate(**args_jump_rate_c_a)

    args_jump_rate_c_f = {"results_barrier_1": results_barrier_fixed_grid_10,
                          "results_barrier_2": results_barrier_cutoff_10,
                          "results_asian_1": results_asian_fixed_grid_10,
                          "results_asian_2": results_asian_cutoff_10,
                          "error_bounds": error_bounds,
                          "identifier": 2}

    create_figures_N_l_jump_rate(**args_jump_rate_c_f)


if __name__ == "__main__":
    plots_main()