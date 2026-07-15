"""
Tools to visualize the results of the convergence rates simulations.
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


def create_figures_exp_err_nc(
        results_barrier_fixed, results_barrier_adapted,
        results_asian_fixed, results_asian_adapted,
        jump_rates):
    """
    This function creates a figure that displays the log(|E[P_l - P_{l-1}]|)-level
    dependency for barrier and asian options for different jump rates.
    Args:
        results_barrier_fixed: This list contains dfs (one for each jump rate)
        that store information about the rate estimation for a barrier option
        using a fixed grid.
        results_barrier_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option using a jump adapted grid.
        results_asian_fixed: This list contains dfs (one for each jump rate)
        that store information about the rate estimation for an asian option
        using a fixed grid.
        results_asian_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option using a jump adapted grid.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier fixed grid
        levels_barrier_fixed = results_barrier_fixed[i].index.values
        mean_error_barrier_fixed = np.log2(results_barrier_fixed[i]["Mean_Error_l"]\
            .to_numpy())
        axs[0, 0].plot(levels_barrier_fixed, mean_error_barrier_fixed,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier jump adapted
        levels_barrier_adapted = results_barrier_adapted[i].index.values
        mean_error_barrier_adapted = \
            np.log2(results_barrier_adapted[i]["Mean_Error_l"].to_numpy())
        axs[1, 0].plot(levels_barrier_adapted, mean_error_barrier_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian fixed grid
        levels_asian_fixed = results_asian_fixed[i].index.values
        mean_error_asian_fixed = np.log2(results_asian_fixed[i]["Mean_Error_l"]\
            .to_numpy())
        axs[0, 1].plot(levels_asian_fixed, mean_error_asian_fixed,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian jump adapted
        levels_asian_adapted = results_asian_adapted[i].index.values
        mean_error_asian_adapted = np.log2(results_asian_adapted[i]["Mean_Error_l"]\
            .to_numpy())
        axs[1, 1].plot(levels_asian_adapted, mean_error_asian_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
    axs[0, 0].set_xlabel(r"$l$")
    axs[0, 0].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    axs[1, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
    axs[1, 0].set_xlabel(r"$l$")
    axs[1, 0].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
    axs[0, 1].set_xlabel(r"$l$")
    axs[0, 1].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")
    axs[1, 1].set_xlabel(r"$l$")
    axs[1, 1].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig("Plots_Estimates/exp_error_nc.pdf", bbox_inches="tight")

    plt.show()


def create_figures_exp_err_c(
        results_barrier_adapted, results_asian_adapted,
        jump_rates):
    """
    This function creates a figure that displays the log(|E[P_l - P_{l-1}]|)-level
    dependency for barrier and asian options for different jump rates.
    Args:
        results_barrier_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option using a jump adapted grid.
        results_asian_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option using a jump adapted grid.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier jump adapted
        levels_barrier_adapted = results_barrier_adapted[i].index.values
        mean_error_barrier_adapted = \
            np.log2(results_barrier_adapted[i]["Mean_Error_l"].to_numpy())
        axs[0].plot(levels_barrier_adapted, mean_error_barrier_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian jump adapted
        levels_asian_adapted = results_asian_adapted[i].index.values
        mean_error_asian_adapted = np.log2(results_asian_adapted[i]["Mean_Error_l"]\
            .to_numpy())
        axs[1].plot(levels_asian_adapted, mean_error_asian_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
    axs[0].set_xlabel(r"$l$")
    axs[0].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    axs[1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
    axs[1].set_xlabel(r"$l$")
    axs[1].set_ylabel(r"$\log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig("Plots_Estimates/exp_error_c.pdf", bbox_inches="tight")

    plt.show()


def create_figures_exp_err_delta(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2, jump_rates,
        identifier):
    """
    This function creates a figure that displays the log(|E[P_l - P_{l-1}]|)-level
    difference between two scheme.
    Args:
        results_barrier_1: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option for the second scheme.
        results_asian_1: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option for the first scheme.
        results_asian_2: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option for the second scheme.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier
        levels_barrier = results_barrier_1[i].index.values
        mean_error_barrier_1 = \
            np.log2(results_barrier_1[i]["Mean_Error_l"].to_numpy())
        mean_error_barrier_2 = \
            np.log2(results_barrier_2[i]["Mean_Error_l"].to_numpy())
        delta_mean_barrier = mean_error_barrier_2 - mean_error_barrier_1
        axs[0].plot(levels_barrier, delta_mean_barrier,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian
        levels_asian = results_asian_1[i].index.values
        mean_error_asian_1 = \
                np.log2(results_asian_1[i]["Mean_Error_l"].to_numpy())
        mean_error_asian_2 = \
            np.log2(results_asian_2[i]["Mean_Error_l"].to_numpy())
        delta_mean_asian = mean_error_asian_2 - mean_error_asian_1
        axs[1].plot(levels_asian, delta_mean_asian,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_xlabel(r"$l$")
    axs[0].set_ylabel(r"$\Delta\, \log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    axs[1].set_xlabel(r"$l$")
    axs[1].set_ylabel(r"$\Delta\, \log_2\; \bigl|\mathbb{E}[P_l - P_{l-1}]\bigr|$")

    if identifier == 0:
        axs[0].set_title("Barrier Option - \nFestgitter - Sprungadaptiert", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nFestgitter - Sprungadaptiert", fontweight="bold")

    elif identifier == 1:
        axs[0].set_title("Barrier Option - \nSprungadaptiert - Cutoff", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nSprungadaptiert - Cutoff", fontweight="bold")

    else:
        axs[0].set_title("Barrier Option - \nFestgitter - Cutoff", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nFestgitter - Cutoff", fontweight="bold")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if identifier == 0:
        plt.savefig("Plots_Estimates/exp_error_delta_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Estimates/exp_error_delta_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Estimates/exp_error_delta_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_var_nc(
        results_barrier_fixed, results_barrier_adapted,
        results_asian_fixed, results_asian_adapted,
        jump_rates):
    """
    This function creates a figure that displays the log(variance)-level dependency
    for barrier and asian options for different jump rates.
    Args:
        results_barrier_fixed: This list contains dfs (one for each jump rate)
        that store information about the rate estimation for a barrier option
        using a fixed grid.
        results_barrier_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option using a jump adapted grid.
        results_asian_fixed: This list contains dfs (one for each jump rate)
        that store information about the rate estimation for an asian option
        using a fixed grid.
        results_asian_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option using a jump adapted grid.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier fixed grid
        levels_barrier_fixed = results_barrier_fixed[i].index.values
        var_barrier_fixed = np.log2(results_barrier_fixed[i]["V_l"] \
                                    .to_numpy())
        axs[0, 0].plot(levels_barrier_fixed, var_barrier_fixed,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier jump adapted
        levels_barrier_adapted = results_barrier_adapted[i].index.values
        var_barrier_adapted = \
            np.log2(results_barrier_adapted[i]["V_l"].to_numpy())
        axs[1, 0].plot(levels_barrier_adapted, var_barrier_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian fixed grid
        levels_asian_fixed = results_asian_fixed[i].index.values
        var_asian_fixed = np.log2(results_asian_fixed[i]["V_l"] \
                                  .to_numpy())
        axs[0, 1].plot(levels_asian_fixed, var_asian_fixed,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian jump adapted
        levels_asian_adapted = results_asian_adapted[i].index.values
        var_asian_adapted = np.log2(results_asian_adapted[i]["V_l"] \
                                    .to_numpy())
        axs[1, 1].plot(levels_asian_adapted, var_asian_adapted,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_title("Barrier Option - Festgitter", fontweight="bold")
    axs[0, 0].set_xlabel(r"$l$")
    axs[0, 0].set_ylabel(r"$\log_2\, V_l$")

    axs[1, 0].set_title("Barrier Option - \nSprungadaptiert", fontweight="bold")
    axs[1, 0].set_xlabel(r"$l$")
    axs[1, 0].set_ylabel(r"$\log_2\, V_l$")

    axs[0, 1].set_title("Asiatische Option - Festgitter", fontweight="bold")
    axs[0, 1].set_xlabel(r"$l$")
    axs[0, 1].set_ylabel(r"$\log_2\, V_l$")

    axs[1, 1].set_title("Asiatische Option - \nSprungadaptiert", fontweight="bold")
    axs[1, 1].set_xlabel(r"$l$")
    axs[1, 1].set_ylabel(r"$\log_2\, V_l$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig("Plots_Estimates/var_nc.pdf", bbox_inches="tight")

    plt.show()


def create_figures_var_c(
        results_barrier_adapted, results_asian_adapted,
        jump_rates):
    """
    This function creates a figure that displays the log(variance)-level dependency
    for barrier and asian options for different jump rates.
    Args:
        results_barrier_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option using a jump adapted grid.
        results_asian_adapted: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option using a jump adapted grid.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier jump adapted
        levels_barrier_adapted = results_barrier_adapted[i].index.values
        var_barrier_adapted = \
            np.log2(results_barrier_adapted[i]["V_l"].to_numpy())
        axs[0].plot(levels_barrier_adapted, var_barrier_adapted,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian jump adapted
        levels_asian_adapted = results_asian_adapted[i].index.values
        var_asian_adapted = np.log2(results_asian_adapted[i]["V_l"] \
                                    .to_numpy())
        axs[1].plot(levels_asian_adapted, var_asian_adapted,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
    axs[0].set_xlabel(r"$l$")
    axs[0].set_ylabel(r"$\log_2\, V_l$")

    axs[1].set_title("Asiatische Option - \nSprungadaptiert (Cutoff)", fontweight="bold")
    axs[1].set_xlabel(r"$l$")
    axs[1].set_ylabel(r"$\log_2\, V_l$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig("Plots_Estimates/var_c.pdf", bbox_inches="tight")

    plt.show()


def create_figures_var_delta(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2, jump_rates,
        identifier):
    """
    This function creates a figure that displays the log(variance)-level difference
    between two scheme.
    Args:
        results_barrier_1: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for a barrier
        option for the second scheme.
        results_asian_1: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option for the first scheme.
        results_asian_2: This list contains dfs (one for each jump
        rate) that store information about the rate estimation for an asian
        option for the second scheme.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier
        levels_barrier = results_barrier_1[i].index.values
        var_barrier_1= \
            np.log2(results_barrier_1[i]["V_l"].to_numpy())
        var_barrier_2 = \
            np.log2(results_barrier_2[i]["V_l"].to_numpy())
        delta_var_barrier = var_barrier_2 - var_barrier_1
        axs[0].plot(levels_barrier, delta_var_barrier,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian
        levels_asian = results_asian_1[i].index.values
        var_asian_1 = \
            np.log2(results_asian_1[i]["V_l"].to_numpy())
        var_asian_2 = np.log2(results_asian_2[i]["V_l"].to_numpy())
        delta_var_asian = var_asian_2 - var_asian_1
        axs[1].plot(levels_asian, delta_var_asian,
                    label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_xlabel(r"$l$")
    axs[0].set_ylabel(r"$\Delta\, \log_2\, V_l$")

    axs[1].set_xlabel(r"$l$")
    axs[1].set_ylabel(r"$\Delta\, \log_2\, V_l$")

    if identifier == 0:
        axs[0].set_title("Barrier Option - \nFestgitter - Sprungadaptiert", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nFestgitter - Sprungadaptiert", fontweight="bold")

    elif identifier == 1:
        axs[0].set_title("Barrier Option - \nSprungadaptiert - Cutoff", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nSprungadaptiert - Cutoff", fontweight="bold")

    else:
        axs[0].set_title("Barrier Option - \nFestgitter - Cutoff", fontweight="bold")
        axs[1].set_title("Asiatische Option - \nFestgitter - Cutoff", fontweight="bold")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_xticks(np.arange(1, 11))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if identifier == 0:
        plt.savefig("Plots_Estimates/var_delta_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Estimates/var_delta_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Estimates/var_delta_c_f.pdf", bbox_inches="tight")

    plt.show()


def plots_main():
    """
    Main function to create and save relevant plots.
    """
    jump_rates = [1, 5, 10, 15, 20]

    # Load results
    results_barrier_fixed_grid = []
    results_barrier_jump_adapted = []
    results_barrier_jump_adapted_cutoff = []
    results_asian_fixed_grid = []
    results_asian_jump_adapted = []
    results_asian_jump_adapted_cutoff = []

    for jump_rate in jump_rates:
        results_df_barrier_fixed = pd.read_feather(
            f"MC_Rates_Estimates_Results/results_MC_barrier_fixed_{jump_rate}.feather")
        results_barrier_fixed_grid.append(
            results_df_barrier_fixed)

        results_df_barrier_adapted = pd.read_feather(
            f"MC_Rates_Estimates_Results/results_MC_barrier_adapted_{jump_rate}.feather")
        results_barrier_jump_adapted.append(
            results_df_barrier_adapted)

        results_df_barrier_adapted_cutoff = pd.read_feather(
             f"MC_Rates_Estimates_Results/results_MC_barrier_cutoff_{jump_rate}.feather")
        results_barrier_jump_adapted_cutoff.append(
            results_df_barrier_adapted_cutoff)

        results_df_asian_fixed = pd.read_feather(
            f"MC_Rates_Estimates_Results/results_MC_asian_fixed_{jump_rate}.feather")
        results_asian_fixed_grid.append(
            results_df_asian_fixed)

        results_df_asian_adapted = pd.read_feather(
            f"MC_Rates_Estimates_Results/results_MC_asian_adapted_{jump_rate}.feather")
        results_asian_jump_adapted.append(
            results_df_asian_adapted)

        results_df_asian_adapted_cutoff = pd.read_feather(
           f"MC_Rates_Estimates_Results/results_MC_asian_cutoff_{jump_rate}.feather")
        results_asian_jump_adapted_cutoff.append(
            results_df_asian_adapted_cutoff)

    args_nc = {"results_barrier_fixed": results_barrier_fixed_grid,
               "results_barrier_adapted": results_barrier_jump_adapted,
               "results_asian_fixed": results_asian_fixed_grid,
               "results_asian_adapted": results_asian_jump_adapted,
               "jump_rates": jump_rates}

    create_figures_exp_err_nc(**args_nc)
    create_figures_var_nc(**args_nc)

    args_c = {"results_barrier_adapted": results_barrier_jump_adapted_cutoff,
              "results_asian_adapted": results_asian_jump_adapted_cutoff,
              "jump_rates": jump_rates}

    create_figures_exp_err_c(**args_c)
    create_figures_var_c(**args_c)

    # Comparison
    args_delta_fixed_nc = {"results_barrier_1": results_barrier_fixed_grid,
                           "results_barrier_2": results_barrier_jump_adapted,
                           "results_asian_1": results_asian_fixed_grid,
                           "results_asian_2": results_asian_jump_adapted,
                           "jump_rates": jump_rates,
                           "identifier": 0}

    create_figures_exp_err_delta(**args_delta_fixed_nc)
    create_figures_var_delta(**args_delta_fixed_nc)

    args_delta_fixed_c_a = {"results_barrier_1": results_barrier_jump_adapted,
                            "results_barrier_2": results_barrier_jump_adapted_cutoff,
                            "results_asian_1": results_asian_jump_adapted,
                            "results_asian_2": results_asian_jump_adapted_cutoff,
                            "jump_rates": jump_rates,
                            "identifier": 1}

    create_figures_exp_err_delta(**args_delta_fixed_c_a)
    create_figures_var_delta(**args_delta_fixed_c_a)

    args_delta_fixed_c_f = {"results_barrier_1": results_barrier_fixed_grid,
                            "results_barrier_2": results_barrier_jump_adapted_cutoff,
                            "results_asian_1": results_asian_fixed_grid,
                            "results_asian_2": results_asian_jump_adapted_cutoff,
                            "jump_rates": jump_rates,
                            "identifier": 2}

    create_figures_exp_err_delta(**args_delta_fixed_c_f)
    create_figures_var_delta(**args_delta_fixed_c_f)


if __name__ == "__main__":
    plots_main()