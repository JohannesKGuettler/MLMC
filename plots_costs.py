"""
Tools to visualize the results for the costs of a MLMC simulation.
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


def create_figures_costs_actual(
        costs_actual_barrier_1, costs_actual_barrier_2,
        costs_actual_asian_1, costs_actual_asian_2,
        error_bounds, jump_rates, identifier):
    """
    This function creates a figure that displays the actual costs-error
    dependency for barrier and asian options for different jump rates.
    Args:
        costs_actual_barrier_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the first scheme.
        costs_actual_barrier_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the second scheme.
        costs_actual_asian_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the first scheme.
        costs_actual_asian_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the second scheme.
        error_bounds: This list contains the error bounds for which
        costs are displayed.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier 1
        actual_costs_barrier_1 = costs_actual_barrier_1[i, :]
        axs[0, 0].plot(error_bounds, actual_costs_barrier_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier 2
        actual_costs_barrier_2 = costs_actual_barrier_2[i, :]
        axs[1, 0].plot(error_bounds, actual_costs_barrier_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 1
        actual_costs_asian_1 = costs_actual_asian_1[i, :]
        axs[0, 1].plot(error_bounds, actual_costs_asian_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 2
        actual_costs_asian_2 = costs_actual_asian_2[i, :]
        axs[1, 1].plot(error_bounds, actual_costs_asian_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_xlabel(r"$\epsilon$")
    axs[0, 0].set_ylabel(r"$C$")

    axs[1, 0].set_xlabel(r"$\epsilon$")
    axs[1, 0].set_ylabel(r"$C$")

    axs[0, 1].set_xlabel(r"$\epsilon$")
    axs[0, 1].set_ylabel(r"$C$")

    axs[1, 1].set_xlabel(r"$\epsilon$")
    axs[1, 1].set_ylabel(r"$C$")

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
        ax.set_xlim((min(error_bounds)-0.005), (max(error_bounds)+0.005))
        ax.set_xticks(error_bounds)
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
        plt.savefig("Plots_Costs/actual_costs_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Costs/actual_costs_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Costs/actual_costs_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_costs_ideal(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2,
        error_bounds, jump_rates, identifier):
    """
    This function creates a figure that displays the idealized costs-error
    dependency for barrier and asian options for different jump rates.
    Args:
        results_barrier_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option
        for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option for
        the second scheme.
        results_asian_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the first scheme.
        results_asian_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the second scheme.
        error_bounds: This list contains the error bounds for which
        costs are displayed.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        ideal_costs_barrier_1 = []
        ideal_costs_barrier_2 = []
        ideal_costs_asian_1 = []
        ideal_costs_asian_2 = []

        for k, bound in enumerate(error_bounds):
            ideal_costs_barrier_1.append(np.dot(results_barrier_1[k][i]["N_l"],
                                                results_barrier_1[k][i]["C_l"]))
            ideal_costs_barrier_2.append(np.dot(results_barrier_2[k][i]["N_l"],
                                                results_barrier_2[k][i]["C_l"]))
            ideal_costs_asian_1.append(np.dot(results_asian_1[k][i]["N_l"],
                                              results_asian_1[k][i]["C_l"]))
            ideal_costs_asian_2.append(np.dot(results_asian_2[k][i]["N_l"],
                                              results_asian_2[k][i]["C_l"]))

        # Barrier 1
        axs[0, 0].plot(error_bounds, ideal_costs_barrier_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier 2
        axs[1, 0].plot(error_bounds, ideal_costs_barrier_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 1
        axs[0, 1].plot(error_bounds, ideal_costs_asian_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 2
        axs[1, 1].plot(error_bounds, ideal_costs_asian_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_xlabel(r"$\epsilon$")
    axs[0, 0].set_ylabel(r"$C\,(idealisiert)$")

    axs[1, 0].set_xlabel(r"$\epsilon$")
    axs[1, 0].set_ylabel(r"$C\,(idealisiert)$")

    axs[0, 1].set_xlabel(r"$\epsilon$")
    axs[0, 1].set_ylabel(r"$C\,(idealisiert)$")

    axs[1, 1].set_xlabel(r"$\epsilon$")
    axs[1, 1].set_ylabel(r"$C\,(idealisiert)$")

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
        ax.set_xlim((min(error_bounds)-0.005), (max(error_bounds)+0.005))
        ax.set_xticks(error_bounds)
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
        plt.savefig("Plots_Costs/ideal_costs_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Costs/ideal_costs_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Costs/ideal_costs_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_costs_actual_ideal_comp(
        costs_actual_barrier_1, costs_actual_barrier_2,
        costs_actual_asian_1, costs_actual_asian_2,
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2,
        error_bounds, jump_rates, identifier):
    """
    This function creates a figure that displays the actual to idealized
    cost difference-error dependency for barrier and asian options for
    different jump rates.
    Args:
        costs_actual_barrier_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the first scheme.
        costs_actual_barrier_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the second scheme.
        costs_actual_asian_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the first scheme.
        costs_actual_asian_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the second scheme.
        results_barrier_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option
        for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option for
        the second scheme.
        results_asian_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the first scheme.
        results_asian_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the second scheme.
        error_bounds: This list contains the error bounds for which
        costs are displayed.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.5))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        ideal_costs_barrier_1 = []
        ideal_costs_barrier_2 = []
        ideal_costs_asian_1 = []
        ideal_costs_asian_2 = []

        for k, bound in enumerate(error_bounds):
            ideal_costs_barrier_1.append(np.dot(results_barrier_1[k][i]["N_l"],
                                                results_barrier_1[k][i]["C_l"]))
            ideal_costs_barrier_2.append(np.dot(results_barrier_2[k][i]["N_l"],
                                                results_barrier_2[k][i]["C_l"]))
            ideal_costs_asian_1.append(np.dot(results_asian_1[k][i]["N_l"],
                                              results_asian_1[k][i]["C_l"]))
            ideal_costs_asian_2.append(np.dot(results_asian_2[k][i]["N_l"],
                                              results_asian_2[k][i]["C_l"]))

        # Barrier 1
        costs_difference_barrier_1 = np.array(costs_actual_barrier_1[i, :]) - \
                                     np.array(ideal_costs_barrier_1)
        axs[0, 0].plot(error_bounds, costs_difference_barrier_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Barrier 2
        costs_difference_barrier_2 = np.array(costs_actual_barrier_2[i, :]) - \
                                     np.array(ideal_costs_barrier_2)
        axs[1, 0].plot(error_bounds, costs_difference_barrier_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 1
        costs_difference_asian_1 =  np.array(costs_actual_asian_1[i, :]) - \
                                    np.array(ideal_costs_asian_1)
        axs[0, 1].plot(error_bounds, costs_difference_asian_1,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian 2
        costs_difference_asian_2 = np.array(costs_actual_asian_2[i, :]) - \
                                   np.array(ideal_costs_asian_2)
        axs[1, 1].plot(error_bounds, costs_difference_asian_2,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0, 0].set_xlabel(r"$\epsilon$")
    axs[0, 0].set_ylabel(r"$C - C\,(idealisiert)$")

    axs[1, 0].set_xlabel(r"$\epsilon$")
    axs[1, 0].set_ylabel(r"$C - C\,(idealisiert)$")

    axs[0, 1].set_xlabel(r"$\epsilon$")
    axs[0, 1].set_ylabel(r"$C - C\,(idealisiert)$")

    axs[1, 1].set_xlabel(r"$\epsilon$")
    axs[1, 1].set_ylabel(r"$C - C\,(idealisiert)$")

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
        ax.set_xlim((min(error_bounds) - 0.005), (max(error_bounds) + 0.005))
        ax.set_xticks(error_bounds)

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
        plt.savefig("Plots_Costs/actual_ideal_costs_comp_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Costs/actual_ideal_costs_comp_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Costs/actual_ideal_costs_comp_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_costs_actual_comp(
        costs_actual_barrier_1, costs_actual_barrier_2,
        costs_actual_asian_1, costs_actual_asian_2,
        error_bounds, jump_rates, identifier):
    """
    This function creates a figure that displays the actual relative
    cost difference-error dependency for barrier and asian options for
    different jump rates.
    Args:
        costs_actual_barrier_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the first scheme.
        costs_actual_barrier_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for a barrier option
        for the second scheme.
        costs_actual_asian_1: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the first scheme.
        costs_actual_asian_2: This list contains arrays (one for each jump rate)
        that store information about the actual costs for an asian option
        for the second scheme.
        error_bounds: This list contains the error bounds for which
        costs are displayed.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        # Barrier
        actual_costs_difference_barrier = \
            (costs_actual_barrier_1[i, :] - costs_actual_barrier_2[i, :]) / costs_actual_barrier_2[i, :]
        axs[0].plot(error_bounds, actual_costs_difference_barrier,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian
        actual_costs_difference_asian = \
            (costs_actual_asian_1[i, :] - costs_actual_asian_2[i, :]) / costs_actual_asian_2[i, :]
        axs[1].plot(error_bounds, actual_costs_difference_asian,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_xlabel(r"$\epsilon$")
    axs[0].set_ylabel(r"$\Delta_{rel}\,C$")

    axs[1].set_xlabel(r"$\epsilon$")
    axs[1].set_ylabel(r"$\Delta_{rel}\,C$")

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
        ax.set_xlim((min(error_bounds)-0.005), (max(error_bounds)+0.005))
        ax.set_xticks(error_bounds)

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
        plt.savefig("Plots_Costs/actual_costs_comp_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Costs/actual_costs_comp_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Costs/actual_costs_comp_c_f.pdf", bbox_inches="tight")

    plt.show()


def create_figures_costs_ideal_comp(
        results_barrier_1, results_barrier_2,
        results_asian_1, results_asian_2,
        error_bounds, jump_rates, identifier):
    """
    This function creates a figure that displays the idealized relative
    cost difference-error dependency for barrier and asian options for
    different jump rates.
    Args:
        results_barrier_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option
        for the first scheme.
        results_barrier_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for a barrier option for
        the second scheme.
        results_asian_1: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the first scheme.
        results_asian_2: This list contains dfs (one for each jump rate)
        that store information about N_l and C_l for an asian option for
        the second scheme.
        error_bounds: This list contains the error bounds for which
        costs are displayed.
        jump_rates: This list contains different values for the jump
        intensity of the underlying jump-diffusion model.
        identifier: This integer specifies what schemes are compared.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    for i, jump_rate in enumerate(jump_rates):
        marker = MARKER[i]

        ideal_costs_barrier_1 = []
        ideal_costs_barrier_2 = []
        ideal_costs_asian_1 = []
        ideal_costs_asian_2 = []

        for k, bound in enumerate(error_bounds):
            ideal_costs_barrier_1.append(np.dot(results_barrier_1[k][i]["N_l"],
                                                results_barrier_1[k][i]["C_l"]))
            ideal_costs_barrier_2.append(np.dot(results_barrier_2[k][i]["N_l"],
                                                results_barrier_2[k][i]["C_l"]))
            ideal_costs_asian_1.append(np.dot(results_asian_1[k][i]["N_l"],
                                              results_asian_1[k][i]["C_l"]))
            ideal_costs_asian_2.append(np.dot(results_asian_2[k][i]["N_l"],
                                              results_asian_2[k][i]["C_l"]))

        # Barrier
        ideal_costs_difference_barrier = \
            (np.array(ideal_costs_barrier_1) - np.array(ideal_costs_barrier_2)) \
            / np.array(ideal_costs_barrier_2)
        axs[0].plot(error_bounds, ideal_costs_difference_barrier,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

        # Asian
        ideal_costs_difference_asian = \
            (np.array(ideal_costs_asian_1) - np.array(ideal_costs_asian_2)) \
            / np.array(ideal_costs_asian_2)
        axs[1].plot(error_bounds, ideal_costs_difference_asian,
                       label=fr"$\lambda = {jump_rate}$", marker=marker)

    axs[0].set_xlabel(r"$\epsilon$")
    axs[0].set_ylabel(r"$\Delta_{rel}\,C\,(idealisiert)$")

    axs[1].set_xlabel(r"$\epsilon$")
    axs[1].set_ylabel(r"$\Delta_{rel}\,C\,(idealisiert)$")

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
        ax.set_xlim((min(error_bounds)-0.005), (max(error_bounds)+0.005))
        ax.set_xticks(error_bounds)

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
        plt.savefig("Plots_Costs/ideal_costs_comp_nc.pdf", bbox_inches="tight")
    elif identifier == 1:
        plt.savefig("Plots_Costs/ideal_costs_comp_c_a.pdf", bbox_inches="tight")
    else:
        plt.savefig("Plots_Costs/ideal_costs_comp_c_f.pdf", bbox_inches="tight")

    plt.show()


def estimate_rates(A, log_costs):
    """
    This function estimates the MLMC convergence rates
    for a given option, scheme and jump rate using least squares.
    Args:
        A: This 2-D array represents the system matrix.
        log_costs: This 1-D array contains the log costs
        for each error bound.

    Returns:
        A float the specifies the estimated convergence rate.
    """
    convergence_rate, _ = np.linalg.lstsq(A, log_costs)[0]

    return convergence_rate


def plots_main():
    """
    Main function to create and save relevant plots.
    """
    jump_rates = [1, 5, 10, 15, 20]
    error_bounds = [0.1, 0.075, 0.05, 0.025]

    # Load cost data
    cost_df_barrier_fixed = pd.read_feather(
        "MLMC_Results/Costs/cost_df_barrier_fixed.feather")
    costs_barrier_fixed_grid = cost_df_barrier_fixed.to_numpy()

    cost_df_barrier_adapted = pd.read_feather(
        "MLMC_Results/Costs/cost_df_barrier_adapted.feather")
    costs_barrier_jump_adapted = cost_df_barrier_adapted.to_numpy()

    cost_df_barrier_cutoff = pd.read_feather(
        "MLMC_Results/Costs/cost_df_barrier_cutoff.feather")
    costs_barrier_cutoff = cost_df_barrier_cutoff.to_numpy()

    cost_df_asian_fixed = pd.read_feather(
        "MLMC_Results/Costs/cost_df_asian_fixed.feather")
    costs_asian_fixed_grid = cost_df_asian_fixed.to_numpy()

    cost_df_asian_adapted = pd.read_feather(
        "MLMC_Results/Costs/cost_df_asian_adapted.feather")
    costs_asian_jump_adapted = cost_df_asian_adapted.to_numpy()

    cost_df_asian_cutoff = pd.read_feather(
        "MLMC_Results/Costs/cost_df_asian_cutoff.feather")
    costs_asian_cutoff = cost_df_asian_cutoff.to_numpy()

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
        results_barrier_jump_adapted.append(
            results_barrier_jump_adapted_temp)
        results_barrier_cutoff.append(results_barrier_cutoff_temp)

        results_asian_fixed_grid.append(results_asian_fixed_grid_temp)
        results_asian_jump_adapted.append(results_asian_jump_adapted_temp)
        results_asian_cutoff.append(results_asian_cutoff_temp)

    # Total costs
    args_actual_nc = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                      "costs_actual_barrier_2": costs_barrier_jump_adapted,
                      "costs_actual_asian_1": costs_asian_fixed_grid,
                      "costs_actual_asian_2": costs_asian_jump_adapted,
                      "error_bounds": error_bounds,
                      "jump_rates": jump_rates,
                      "identifier": 0}

    create_figures_costs_actual(**args_actual_nc)

    args_actual_c_a = {"costs_actual_barrier_1": costs_barrier_jump_adapted,
                       "costs_actual_barrier_2": costs_barrier_cutoff,
                       "costs_actual_asian_1": costs_asian_jump_adapted,
                       "costs_actual_asian_2": costs_asian_cutoff,
                       "error_bounds": error_bounds,
                       "jump_rates": jump_rates,
                       "identifier": 1}

    create_figures_costs_actual(**args_actual_c_a)

    args_actual_c_f = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                       "costs_actual_barrier_2": costs_barrier_cutoff,
                       "costs_actual_asian_1": costs_asian_fixed_grid,
                       "costs_actual_asian_2": costs_asian_cutoff,
                       "error_bounds": error_bounds,
                       "jump_rates": jump_rates,
                       "identifier": 2}

    create_figures_costs_actual(**args_actual_c_f)

    args_ideal_nc = {"results_barrier_1": results_barrier_fixed_grid,
                     "results_barrier_2": results_barrier_jump_adapted,
                     "results_asian_1": results_asian_fixed_grid,
                     "results_asian_2": results_asian_jump_adapted,
                     "error_bounds": error_bounds,
                     "jump_rates": jump_rates,
                     "identifier": 0}

    create_figures_costs_ideal(**args_ideal_nc)

    args_ideal_c_a = {"results_barrier_1": results_barrier_jump_adapted,
                      "results_barrier_2": results_barrier_cutoff,
                      "results_asian_1": results_asian_jump_adapted,
                      "results_asian_2": results_asian_cutoff,
                      "error_bounds": error_bounds,
                      "jump_rates": jump_rates,
                      "identifier": 1}

    create_figures_costs_ideal(**args_ideal_c_a)

    args_ideal_c_f = {"results_barrier_1": results_barrier_fixed_grid,
                      "results_barrier_2": results_barrier_cutoff,
                      "results_asian_1": results_asian_fixed_grid,
                      "results_asian_2": results_asian_cutoff,
                      "error_bounds": error_bounds,
                      "jump_rates": jump_rates,
                      "identifier": 2}

    create_figures_costs_ideal(**args_ideal_c_f)

    # Actual vs. idealized
    args_actual_ideal_comp_nc = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                                 "costs_actual_barrier_2": costs_barrier_jump_adapted,
                                 "costs_actual_asian_1": costs_asian_fixed_grid,
                                 "costs_actual_asian_2": costs_asian_jump_adapted,
                                 "results_barrier_1": results_barrier_fixed_grid,
                                 "results_barrier_2": results_barrier_jump_adapted,
                                 "results_asian_1": results_asian_fixed_grid,
                                 "results_asian_2": results_asian_jump_adapted,
                                 "error_bounds": error_bounds,
                                 "jump_rates": jump_rates,
                                 "identifier": 0}

    create_figures_costs_actual_ideal_comp(**args_actual_ideal_comp_nc)

    args_actual_ideal_comp_c_a = {"costs_actual_barrier_1": costs_barrier_jump_adapted,
                                  "costs_actual_barrier_2": costs_barrier_cutoff,
                                  "costs_actual_asian_1": costs_asian_jump_adapted,
                                  "costs_actual_asian_2": costs_asian_cutoff,
                                  "results_barrier_1": results_barrier_jump_adapted,
                                  "results_barrier_2": results_barrier_cutoff,
                                  "results_asian_1": results_asian_jump_adapted,
                                  "results_asian_2": results_asian_cutoff,
                                  "error_bounds": error_bounds,
                                  "jump_rates": jump_rates,
                                  "identifier": 1}

    create_figures_costs_actual_ideal_comp(**args_actual_ideal_comp_c_a)

    args_actual_ideal_comp_c_f = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                                  "costs_actual_barrier_2": costs_barrier_cutoff,
                                  "costs_actual_asian_1": costs_asian_fixed_grid,
                                  "costs_actual_asian_2": costs_asian_cutoff,
                                  "results_barrier_1": results_barrier_fixed_grid,
                                  "results_barrier_2": results_barrier_cutoff,
                                  "results_asian_1": results_asian_fixed_grid,
                                  "results_asian_2": results_asian_cutoff,
                                  "error_bounds": error_bounds,
                                  "jump_rates": jump_rates,
                                  "identifier": 2}

    create_figures_costs_actual_ideal_comp(**args_actual_ideal_comp_c_f)

    # Comparison
    args_actual_comp_nc = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                           "costs_actual_barrier_2": costs_barrier_jump_adapted,
                           "costs_actual_asian_1": costs_asian_fixed_grid,
                           "costs_actual_asian_2": costs_asian_jump_adapted,
                           "error_bounds": error_bounds,
                           "jump_rates": jump_rates,
                           "identifier": 0}

    create_figures_costs_actual_comp(**args_actual_comp_nc)

    args_actual_comp_c_a = {"costs_actual_barrier_1": costs_barrier_jump_adapted,
                            "costs_actual_barrier_2": costs_barrier_cutoff,
                            "costs_actual_asian_1": costs_asian_jump_adapted,
                            "costs_actual_asian_2": costs_asian_cutoff,
                            "error_bounds": error_bounds,
                            "jump_rates": jump_rates,
                            "identifier": 1}

    create_figures_costs_actual_comp(**args_actual_comp_c_a)

    args_actual_comp_c_f = {"costs_actual_barrier_1": costs_barrier_fixed_grid,
                            "costs_actual_barrier_2": costs_barrier_cutoff,
                            "costs_actual_asian_1": costs_asian_fixed_grid,
                            "costs_actual_asian_2": costs_asian_cutoff,
                            "error_bounds": error_bounds,
                            "jump_rates": jump_rates,
                            "identifier": 2}

    create_figures_costs_actual_comp(**args_actual_comp_c_f)

    args_ideal_comp_nc = {"results_barrier_1": results_barrier_fixed_grid,
                          "results_barrier_2": results_barrier_jump_adapted,
                          "results_asian_1": results_asian_fixed_grid,
                          "results_asian_2": results_asian_jump_adapted,
                          "error_bounds": error_bounds,
                          "jump_rates": jump_rates,
                          "identifier": 0}

    create_figures_costs_ideal_comp(**args_ideal_comp_nc)

    args_ideal_comp_c_a = {"results_barrier_1": results_barrier_jump_adapted,
                           "results_barrier_2": results_barrier_cutoff,
                           "results_asian_1": results_asian_jump_adapted,
                           "results_asian_2": results_asian_cutoff,
                           "error_bounds": error_bounds,
                           "jump_rates": jump_rates,
                           "identifier": 1}

    create_figures_costs_ideal_comp(**args_ideal_comp_c_a)

    args_ideal_comp_c_f = {"results_barrier_1": results_barrier_fixed_grid,
                           "results_barrier_2": results_barrier_cutoff,
                           "results_asian_1": results_asian_fixed_grid,
                           "results_asian_2": results_asian_cutoff,
                           "error_bounds": error_bounds,
                           "jump_rates": jump_rates,
                           "identifier": 2}

    create_figures_costs_ideal_comp(**args_ideal_comp_c_f)

    # Estimate overall convergence rate
    A = np.column_stack((np.log2(np.array(error_bounds)),
                         np.ones_like(np.array(error_bounds))))
    columns = ["Barrier Fixed Grid", "Asian Fixed Grid",
               "Barrier Jump Adapted", "Asian Jump Adapted",
               "Barrier Cutoff", "Asian Cutoff"]

    # Actual costs
    convergence_rates_actual_df = pd.DataFrame(columns=columns,
                                               index=np.array(jump_rates))
    for i, jump_rate in enumerate(jump_rates):
        # Fixed Grid
        convergence_rates_actual_df.loc[jump_rate, "Barrier Fixed Grid"] =\
            estimate_rates(A, np.log2(costs_barrier_fixed_grid[i, :]))
        convergence_rates_actual_df.loc[jump_rate, "Asian Fixed Grid"] = \
            estimate_rates(A, np.log2(costs_asian_fixed_grid[i, :]))

        # Jump adapted
        convergence_rates_actual_df.loc[jump_rate, "Barrier Jump Adapted"] = \
            estimate_rates(A, np.log2(costs_barrier_jump_adapted[i, :]))
        convergence_rates_actual_df.loc[jump_rate, "Asian Jump Adapted"] = \
            estimate_rates(A, np.log2(costs_asian_jump_adapted[i, :]))

        # Cutoff
        convergence_rates_actual_df.loc[jump_rate, "Barrier Cutoff"] = \
            estimate_rates(A, np.log2(costs_barrier_cutoff[i, :]))
        convergence_rates_actual_df.loc[jump_rate, "Asian Cutoff"] = \
            estimate_rates(A, np.log2(costs_asian_cutoff[i, :]))

    # Idealized costs
    convergence_rates_ideal_df = pd.DataFrame(columns=columns,
                                               index=np.array(jump_rates))

    for i, jump_rate in enumerate(jump_rates):
        # Calculate ideal costs
        ideal_costs_barrier_fixed = []
        ideal_costs_barrier_adapted = []
        ideal_costs_barrier_cutoff = []

        ideal_costs_asian_fixed = []
        ideal_costs_asian_adapted = []
        ideal_costs_asian_cutoff = []

        for k, bound in enumerate(error_bounds):
            ideal_costs_barrier_fixed.append(
                np.dot(results_barrier_fixed_grid[k][i]["N_l"],
                       results_barrier_fixed_grid[k][i]["C_l"]))
            ideal_costs_barrier_adapted.append(
                np.dot(results_barrier_fixed_grid[k][i]["N_l"],
                       results_barrier_fixed_grid[k][i]["C_l"]))
            ideal_costs_barrier_cutoff.append(
                np.dot(results_barrier_cutoff[k][i]["N_l"],
                       results_barrier_cutoff[k][i]["C_l"]))

            ideal_costs_asian_fixed.append(
                np.dot(results_asian_fixed_grid[k][i]["N_l"],
                       results_asian_fixed_grid[k][i]["C_l"]))
            ideal_costs_asian_adapted.append(
                np.dot(results_asian_jump_adapted[k][i]["N_l"],
                       results_asian_jump_adapted[k][i]["C_l"]))
            ideal_costs_asian_cutoff.append(
                np.dot(results_asian_cutoff[k][i]["N_l"],
                       results_asian_cutoff[k][i]["C_l"]))

        # Fixed Grid
        convergence_rates_ideal_df.loc[jump_rate, "Barrier Fixed Grid"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_barrier_fixed)))
        convergence_rates_ideal_df.loc[jump_rate, "Asian Fixed Grid"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_asian_fixed)))

        # Jump adapted
        convergence_rates_ideal_df.loc[jump_rate, "Barrier Jump Adapted"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_barrier_adapted)))
        convergence_rates_ideal_df.loc[jump_rate, "Asian Jump Adapted"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_asian_adapted)))

        # Cutoff
        convergence_rates_ideal_df.loc[jump_rate, "Barrier Cutoff"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_barrier_fixed)))
        convergence_rates_ideal_df.loc[jump_rate, "Asian Cutoff"] = \
            estimate_rates(A, np.log2(np.array(ideal_costs_asian_cutoff)))

    convergence_rates_actual_df.to_feather("Rates_Estimates/convergence_rate_actual.feather")
    convergence_rates_ideal_df.to_feather("Rates_Estimates/convergence_rate_ideal.feather")


if __name__ == "__main__":
    plots_main()