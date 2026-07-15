"""
Tools to visualize the results for the obtained prices from a MLMC simulation.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})


def create_figures_single_comparison(prices_barrier_fixed, prices_barrier_adapted,
                                     prices_barrier_cutoff, prices_asian_fixed,
                                     prices_asian_adapted, prices_asian_cutoff,
                                     prices_mc_benchmark, error_bounds,
                                     jump_rates):
    """
    This function creates a figure for each provided jump rate that compares
    the MLMC option prices with a MC benchmark price for different error
    bounds.
    Args:
        prices_barrier_fixed: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a fixed grid. (axis=0:
        jump rates, axis=1: error bounds).
        prices_barrier_adapted: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a jump adapted grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_barrier_cutoff: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a jump adapted
        cutoff grid (axis=0: jump rates, axis=1: error bounds).
        prices_asian_fixed: This df contains the obtained prices for an
        asian option from the MLMC simulations using a fixed grid. (axis=0:
        jump rates, axis=1: error bounds).
        prices_asian_adapted: This df contains the obtained prices for an
        asian option from the MLMC simulations using a jump adapted grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_asian_cutoff: This df contains the obtained prices for an
        asian option from the MLMC simulations using a jump adapted
        cutoff grid (axis=0: jump rates, axis=1: error bounds).
        prices_mc_benchmark: This df contains the obtained prices for a barrier
        and an asian option from a MC simulation (N=1e6, M=1e3) where each row
        corresponds to a different jump rate.
        error_bounds: This 1-D array contains different error bounds at which
        convergence is assumed to be achieved.
        jump_rates: This 1-D array contains different values for the jump
        intensity of the underlying jump-diffusion model.
    """
    for i, jump_rate in enumerate(jump_rates):
        fig, axs = plt.subplots(3, 2, figsize=(6.3, 9.8))

        mc_barrier_fixed = prices_mc_benchmark.loc[jump_rate, "Barrier Fixed Grid"]
        mc_barrier_adapted = prices_mc_benchmark.loc[jump_rate, "Barrier Jump Adapted"]
        mc_barrier_cutoff = prices_mc_benchmark.loc[jump_rate, "Barrier Cutoff"]
        mc_asian_fixed = prices_mc_benchmark.loc[jump_rate, "Asian Fixed Grid"]
        mc_asian_adapted = prices_mc_benchmark.loc[jump_rate, "Asian Jump Adapted"]
        mc_asian_cutoff = prices_mc_benchmark.loc[jump_rate, "Asian Cutoff"]

        # Barrier
        axs[0, 0].plot(error_bounds,
                       prices_barrier_fixed.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[0, 0].axhline(y=mc_barrier_fixed, color="red", linestyle="--",
                          label="MC")

        axs[1, 0].plot(error_bounds,
                       prices_barrier_adapted.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[1, 0].axhline(y=mc_barrier_adapted, color="red", linestyle="--",
                          label="MC")

        axs[2, 0].plot(error_bounds,
                       prices_barrier_cutoff.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[2, 0].axhline(y=mc_barrier_cutoff, color="red", linestyle="--",
                          label="MC")

        # Asian
        axs[0, 1].plot(error_bounds,
                       prices_asian_fixed.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[0, 1].axhline(y=mc_asian_fixed, color="red", linestyle="--",
                          label="MC")

        axs[1, 1].plot(error_bounds,
                       prices_asian_adapted.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[1, 1].axhline(y=mc_asian_adapted, color="red", linestyle="--",
                          label="MC")

        axs[2, 1].plot(error_bounds,
                       prices_asian_cutoff.loc[jump_rate, :].to_numpy(),
                       label="MLMC", marker="x")
        axs[2, 1].axhline(y=mc_asian_cutoff, color="red", linestyle="--",
                          label="MC")

        axs[0, 0].set_title("Barrier Option - Festgitter")
        axs[0, 0].set_xlabel(r"$\epsilon$")
        axs[0, 0].set_ylabel(r"$Price$")

        axs[1, 0].set_title("Barrier Option - \nSprungadaptiert")
        axs[1, 0].set_xlabel(r"$\epsilon$")
        axs[1, 0].set_ylabel(r"$Price$")

        axs[2, 0].set_title("Barrier Option - \nSprungadaptiert (Cutoff)")
        axs[2, 0].set_xlabel(r"$\epsilon$")
        axs[2, 0].set_ylabel(r"$Price$")

        axs[0, 1].set_title("Asian Option - Festgitter")
        axs[0, 1].set_xlabel(r"$\epsilon$")
        axs[0, 1].set_ylabel(r"$Price$")

        axs[1, 1].set_title("Asian Option - \nSprungadaptiert")
        axs[1, 1].set_xlabel(r"$\epsilon$")
        axs[1, 1].set_ylabel(r"$Price$")

        axs[2, 1].set_title("Asian Option - \nSprungadaptiert (Cutoff)")
        axs[2, 1].set_xlabel(r"$\epsilon$")
        axs[2, 1].set_ylabel(r"$Price$")

        for ax in axs.flat:
            ax.grid(True, linestyle="-", alpha=0.3)

        handles, labels = axs[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            frameon=True
        )

        fig.suptitle(fr"$\lambda = {jump_rate}$", fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        # fig.tight_layout(rect=(0, 0.08, 1, 1))

        plt.show()
        path = f"Plots_Prices/Price_MC_MLMC_{jump_rate}.pdf"
        fig.savefig(path)


def create_figures_fixed_adapted_comparison(prices_barrier_fixed, prices_barrier_adapted,
                                            prices_asian_fixed, prices_asian_adapted,
                                            prices_mc_benchmark, error_bounds):
    """
    This function creates a figure for jump rate = 10 that compares
    the MLMC fixed grid and jump adapted option prices with a
    MC benchmark price for different error bounds.
    Args:
        prices_barrier_fixed: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a fixed grid. (axis=0:
        jump rates, axis=1: error bounds).
        prices_barrier_adapted: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a jump adapted grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_asian_fixed: This df contains the obtained prices for an
        asian option from the MLMC simulations using a fixed grid. (axis=0:
        jump rates, axis=1: error bounds).
        prices_asian_adapted: This df contains the obtained prices for an
        asian option from the MLMC simulations using a jump adapted grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_mc_benchmark: This df contains the obtained prices for a barrier
        and an asian option from a MC simulation (N=1e6, M=1e3) where each row
        corresponds to a different jump rate.
        error_bounds: This 1-D array contains different error bounds at which
        convergence is assumed to be achieved.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    mc_barrier_adapted = prices_mc_benchmark.loc[10, "Barrier Jump Adapted"]
    mc_asian_adapted = prices_mc_benchmark.loc[10, "Asian Jump Adapted"]

    # Barrier
    axs[0].plot(error_bounds, prices_barrier_fixed.loc[10, :].to_numpy(),
                label="MLMC Festgitter", marker="x")
    axs[0].plot(error_bounds, prices_barrier_adapted.loc[10, :].to_numpy(),
                label="MLMC Sprungadaptiert", marker="o")
    axs[0].axhline(y=mc_barrier_adapted, color="red", linestyle="--",
                   label="MC")

    # Asian
    axs[1].plot(error_bounds, prices_asian_fixed.loc[10, :].to_numpy(),
                label="MLMC Festgitter", marker="x")
    axs[1].plot(error_bounds, prices_asian_adapted.loc[10, :].to_numpy(),
                label="MLMC Sprungadaptiert", marker="o")
    axs[1].axhline(y=mc_asian_adapted, color="red", linestyle="--",
                   label="MC")

    axs[0].set_title("Barrier Option")
    axs[0].set_xlabel(r"$\epsilon$")
    axs[0].set_ylabel(r"$Price$")

    axs[1].set_title("Asiatische Option")
    axs[1].set_xlabel(r"$\epsilon$")
    axs[1].set_ylabel(r"$Price$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.show()
    path = f"Plots_Prices/Price_MC_MLMC_cutoff.pdf"
    fig.savefig(path)


def create_figures_cutoff_comparison(prices_barrier_cutoff, prices_asian_cutoff,
                                     prices_mc_benchmark, error_bounds):
    """
    This function creates a figure for jump rate = 10 that compares
    the MLMC jump adapted cutoff option prices with a
    MC benchmark price for different error bounds.
    Args:
        prices_barrier_cutoff: This df contains the obtained prices for a
        barrier option from the MLMC simulations using a jump adapted cutoff grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_asian_cutoff: This df contains the obtained prices for an
        asian option from the MLMC simulations using a jump adapted cutoff grid.
        (axis=0: jump rates, axis=1: error bounds).
        prices_mc_benchmark: This df contains the obtained prices for a barrier
        and an asian option from a MC simulation (N=1e6, M=1e3) where each row
        corresponds to a different jump rate.
        error_bounds: This 1-D array contains different error bounds at which
        convergence is assumed to be achieved.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.2))

    mc_barrier_cutoff = prices_mc_benchmark.loc[10, "Barrier Jump Adapted"]
    mc_asian_cutoff = prices_mc_benchmark.loc[10, "Asian Jump Adapted"]

    # Barrier
    axs[0].plot(error_bounds, prices_barrier_cutoff.loc[10, :].to_numpy(),
                label="MLMC Sprungadaptiert (Cutoff)", marker="x")
    axs[0].axhline(y=mc_barrier_cutoff, color="red", linestyle="--",
                   label="MC")

    # Asian
    axs[1].plot(error_bounds, prices_asian_cutoff.loc[10, :].to_numpy(),
                label="MLMC Sprungadaptiert (Cutoff)", marker="x")
    axs[1].axhline(y=mc_asian_cutoff, color="red", linestyle="--",
                   label="MC")

    axs[0].set_title("Barrier Option")
    axs[0].set_xlabel(r"$\epsilon$")
    axs[0].set_ylabel(r"$Price$")

    axs[1].set_title("Asiatische Option")
    axs[1].set_xlabel(r"$\epsilon$")
    axs[1].set_ylabel(r"$Price$")

    for ax in axs.flat:
        ax.grid(True, linestyle="-", alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plt.show()
    path = f"Plots_Prices/Price_MC_MLMC_cutoff.pdf"
    fig.savefig(path)


def plots_main():
    """
    Main function to create and save relevant plots.
    """
    jump_rates = [1, 5, 10, 15, 20]
    error_bounds = [0.1, 0.075, 0.05, 0.025]

    # Load price data
    mc_prices = pd.read_feather("MLMC_Results/Prices/mc_benchmark_results.feather")

    prices_df_barrier_fixed = pd.read_feather("MLMC_Results/Prices/prices_df_barrier_fixed.feather")
    prices_df_barrier_adapted = pd.read_feather("MLMC_Results/Prices/prices_df_barrier_adapted.feather")
    prices_df_barrier_cutoff = pd.read_feather("MLMC_Results/Prices/prices_df_barrier_cutoff.feather")

    prices_df_asian_fixed = pd.read_feather("MLMC_Results/Prices/prices_df_asian_fixed.feather")
    prices_df_asian_adapted = pd.read_feather("MLMC_Results/Prices/prices_df_asian_adapted.feather")
    prices_df_asian_cutoff = pd.read_feather("MLMC_Results/Prices/prices_df_asian_cutoff.feather")

    args_single_comp = {"prices_barrier_fixed": prices_df_barrier_fixed,
                        "prices_barrier_adapted": prices_df_barrier_adapted,
                        "prices_barrier_cutoff": prices_df_barrier_cutoff,
                        "prices_asian_fixed": prices_df_asian_fixed,
                        "prices_asian_adapted": prices_df_asian_adapted,
                        "prices_asian_cutoff": prices_df_asian_cutoff,
                        "prices_mc_benchmark": mc_prices,
                        "error_bounds": error_bounds,
                        "jump_rates":jump_rates}

    create_figures_single_comparison(**args_single_comp)

    args_fixed_adapted_comp = {"prices_barrier_fixed": prices_df_barrier_fixed,
                               "prices_barrier_adapted": prices_df_barrier_adapted,
                               "prices_asian_fixed": prices_df_asian_fixed,
                               "prices_asian_adapted": prices_df_asian_adapted,
                               "prices_mc_benchmark": mc_prices,
                               "error_bounds": error_bounds}

    create_figures_fixed_adapted_comparison(**args_fixed_adapted_comp)

    args_cutoff_comp = {"prices_barrier_cutoff": prices_df_barrier_cutoff,
                        "prices_asian_cutoff": prices_df_asian_cutoff,
                        "prices_mc_benchmark": mc_prices,
                        "error_bounds": error_bounds}

    create_figures_cutoff_comparison(**args_cutoff_comp)


if __name__ == "__main__":
    plots_main()