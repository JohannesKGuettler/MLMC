"""
Tools to price options with MLMC.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import pandas as pd
import numpy as np
import mlmc


def g_inverse(y, jump_rate, jumps_mean, jumps_std):
    """
    This function calculates a cutoff value below which jumps are not simulated.
    Args:
        y: An integer that specifies the number of fixed grid points (2^l).
        jump_rate: A float that specifies the jump rate.
        jumps_mean: A float that specifies the mean jump size.
        jumps_std: A float that specifies the standard deviation of the jump size.

    Returns:
        A float that specifies the jump cutoff value.
    """
    return np.sqrt(((jumps_mean ** 2 + jumps_std ** 2) * jump_rate) / y)


def main():
    # Arguments and model initialization
    r = 0.1
    vola = 0.6
    jumps_mean = 0.1
    jumps_std = 0.2

    # Options
    args_barrier = {"S_0": 100,
                    "strike": 80,
                    "maturity": 1,
                    "r": r,
                    "barrier": 50,
                    "up": False}
    args_asian = {"S_0": 100,
                  "strike": 80,
                  "maturity": 1,
                  "r": r,
                  "call": True}

    error_bounds = np.array([0.1, 0.075, 0.05, 0.025])
    jump_rates = np.array([1, 5, 10, 15, 20])

    costs_barrier_fixed_grid = []
    costs_barrier_jump_adapted = []
    costs_barrier_adapted_cutoff = []

    costs_asian_fixed_grid = []
    costs_asian_jump_adapted = []
    costs_asian_adapted_cutoff = []

    prices_barrier_fixed = []
    prices_barrier_adapted = []
    prices_barrier_cutoff = []

    prices_asian_fixed = []
    prices_asian_adapted = []
    prices_asian_cutoff = []

    # Initialize mlmc parameters
    initial_samples = {"Barrier Fixed": [10000] * 3,
                       "Barrier Adapted": [10000] * 3,
                       "Barrier Cutoff": [10000] * 3,
                       "Asian Fixed": [10000] * 3,
                       "Asian Adapted": [10000] * 3,
                       "Asian Cutoff": [10000] * 3}

    rates_barrier_fixed_grid = {"alpha": [0, True],
                                "beta": [0, True],
                                "theta": [1, True]}
    rates_barrier_jump_adapted = {"alpha": [1, True],
                                  "beta": [0, True],
                                  "theta": [1, True]}
    rates_barrier_cutoff = {"alpha": [1, True],
                            "beta": [0, True],
                            "theta": [1, True]}

    rates_asian_fixed_grid = {"alpha": [1, True],
                              "beta": [0, True],
                              "theta": [1, True]}
    rates_asian_jump_adapted = {"alpha": [1, True],
                                "beta": [2, True],
                                "theta": [1, True]}
    rates_asian_cutoff = {"alpha": [1, True],
                          "beta": [2, True],
                          "theta": [1, True]}

    alpha_estimates = pd.read_feather("Rates_Estimates/MC_rates_estimates_alpha.feather")
    beta_estimates = pd.read_feather("Rates_Estimates/MC_rates_estimates_beta.feather")

    for i, bound in enumerate(error_bounds):
        costs_barrier_fixed_grid_temp = []
        costs_barrier_jump_adapted_temp = []
        costs_barrier_cutoff_temp = []

        costs_asian_fixed_grid_temp = []
        costs_asian_jump_adapted_temp = []
        costs_asian_cutoff_temp = []

        prices_barrier_fixed_temp = []
        prices_barrier_adapted_temp = []
        prices_barrier_cutoff_temp = []

        prices_asian_fixed_temp = []
        prices_asian_adapted_temp = []
        prices_asian_cutoff_temp = []

        for jump_rate in jump_rates:
            # Update rates
            # Alpha
            rates_barrier_fixed_grid["alpha"][0] = alpha_estimates["Barrier Fixed Grid"].loc[jump_rate]
            rates_barrier_cutoff["alpha"][0] = alpha_estimates["Barrier Cutoff"].loc[jump_rate]

            rates_asian_cutoff["alpha"][0] = alpha_estimates["Asian Cutoff"].loc[jump_rate]

            # Beta
            rates_barrier_fixed_grid["beta"][0] = beta_estimates["Barrier Fixed Grid"].loc[jump_rate]
            rates_barrier_jump_adapted["beta"][0] = beta_estimates["Barrier Jump Adapted"].loc[jump_rate]
            rates_barrier_cutoff["beta"][0] = beta_estimates["Barrier Cutoff"].loc[jump_rate]

            rates_asian_fixed_grid["beta"][0] = beta_estimates["Asian Fixed Grid"].loc[jump_rate]
            rates_asian_cutoff["beta"][0] = beta_estimates["Asian Cutoff"].loc[jump_rate]

            # Define jump model parameters
            args_merton = {"r": r,
                           "vola": vola,
                           "jump_rate": jump_rate,
                           "jumps_mean": jumps_mean,
                           "jumps_std": jumps_std}

            # Barrier option
            print("##################################")
            print(f"barrier fixed grid, error: {bound}, jump rate: {jump_rate}")
            args_barrier_fixed = {"N_0": initial_samples["Barrier Fixed"],
                                  "L": len(initial_samples["Barrier Fixed"]) - 1,
                                  "rates": rates_barrier_fixed_grid,
                                  "error_bound": bound,
                                  "jm_args": args_merton,
                                  "option_args": args_barrier,
                                  "option_identifier": "Barrier",
                                  "fixed_grid": True,
                                  "cutoff_function": None,
                                  "option_name": "barrier fixed"}
            results_df_barrier_fixed, costs_barrier_fixed, price_barrier_fixed = \
                mlmc.mlmc_main(**args_barrier_fixed)

            costs_barrier_fixed_grid_temp.append(costs_barrier_fixed)
            prices_barrier_fixed_temp.append(price_barrier_fixed)

            print("##################################")
            print(f"barrier jump adapted, error: {bound}, jump rate: {jump_rate}")
            args_barrier_adapted = {"N_0": initial_samples["Barrier Adapted"],
                                    "L": len(initial_samples["Barrier Adapted"]) - 1,
                                    "rates": rates_barrier_jump_adapted,
                                    "error_bound": bound,
                                    "jm_args": args_merton,
                                    "option_args": args_barrier,
                                    "option_identifier": "Barrier",
                                    "fixed_grid": False,
                                    "cutoff_function": None,
                                    "option_name": "barrier adapted"}
            results_df_barrier_adapted, costs_barrier_adapted, price_barrier_adapted = \
                mlmc.mlmc_main(**args_barrier_adapted)

            costs_barrier_jump_adapted_temp.append(costs_barrier_adapted)
            prices_barrier_adapted_temp.append(price_barrier_adapted)

            print("##################################")
            print(f"barrier cutoff, error: {bound}, jump rate: {jump_rate}")
            args_barrier_cutoff = {"N_0": initial_samples["Barrier Cutoff"],
                                  "L": len(initial_samples["Barrier Cutoff"]) - 1,
                                  "rates": rates_barrier_cutoff,
                                  "error_bound": bound,
                                  "jm_args": args_merton,
                                  "option_args": args_barrier,
                                  "option_identifier": "Barrier",
                                  "fixed_grid": False,
                                  "cutoff_function": g_inverse,
                                  "option_name": "barrier cutoff"}
            results_df_barrier_cutoff, costs_barrier_cutoff, price_barrier_cutoff= \
                mlmc.mlmc_main(**args_barrier_cutoff)

            costs_barrier_cutoff_temp.append(costs_barrier_cutoff)
            prices_barrier_cutoff_temp.append(price_barrier_cutoff)

            # Asian option
            print("##################################")
            print(f"asian fixed grid, error: {bound}, jump rate: {jump_rate}")
            args_asian_fixed = {"N_0": initial_samples["Asian Fixed"],
                                  "L": len(initial_samples["Asian Fixed"]) - 1,
                                  "rates": rates_asian_fixed_grid,
                                  "error_bound": bound,
                                  "jm_args": args_merton,
                                  "option_args": args_asian,
                                  "option_identifier": "Asian",
                                  "fixed_grid": True,
                                  "cutoff_function": None,
                                  "option_name": "asian fixed"}
            results_df_asian_fixed, costs_asian_fixed, price_asian_fixed = \
                mlmc.mlmc_main(**args_asian_fixed)

            costs_asian_fixed_grid_temp.append(costs_asian_fixed)
            prices_asian_fixed_temp.append(price_asian_fixed)

            print("##################################")
            print(f"asian jump adapted, error: {bound}, jump rate: {jump_rate}")
            args_asian_adapted = {"N_0": initial_samples["Asian Adapted"],
                                  "L": len(initial_samples["Asian Adapted"]) - 1,
                                  "rates": rates_asian_jump_adapted,
                                  "error_bound": bound,
                                  "jm_args": args_merton,
                                  "option_args": args_asian,
                                  "option_identifier": "Asian",
                                  "fixed_grid": False,
                                  "cutoff_function": None,
                                  "option_name": "asian adapted"}
            results_df_asian_adapted, costs_asian_adapted, price_asian_adapted = \
                mlmc.mlmc_main(**args_asian_adapted)

            costs_asian_jump_adapted_temp.append(costs_asian_adapted)
            prices_asian_adapted_temp.append(price_asian_adapted)

            print("##################################")
            print(f"asian cutoff, error: {bound}, jump rate: {jump_rate}")
            args_asian_cutoff = {"N_0": initial_samples["Asian Cutoff"],
                                 "L": len(initial_samples["Asian Cutoff"]) - 1,
                                 "rates": rates_asian_cutoff,
                                 "error_bound": bound,
                                 "jm_args": args_merton,
                                 "option_args": args_asian,
                                 "option_identifier": "Asian",
                                 "fixed_grid": False,
                                 "cutoff_function": g_inverse,
                                 "option_name": "asian cutoff"}
            results_df_asian_cutoff, costs_asian_cutoff, price_asian_cutoff = \
                mlmc.mlmc_main(**args_asian_cutoff)

            costs_asian_cutoff_temp.append(costs_asian_cutoff)
            prices_asian_cutoff_temp.append(price_asian_cutoff)

            # Save results (C_l, N_l, V_l, |E[P_l - P_{l-1}]|)
            results_df_barrier_fixed.to_feather(
                f"MLMC_Results/Results/results_barrier_fixed_{bound}_{jump_rate}.feather")
            results_df_barrier_adapted.to_feather(
                f"MLMC_Results/Results/results_barrier_adapted_{bound}_{jump_rate}.feather")
            results_df_barrier_cutoff.to_feather(
                f"MLMC_Results/Results/results_barrier_cutoff_{bound}_{jump_rate}.feather")

            results_df_asian_fixed.to_feather(
                f"MLMC_Results/Results/results_asian_fixed_{bound}_{jump_rate}.feather")
            results_df_asian_adapted.to_feather(
                f"MLMC_Results/Results/results_asian_adapted_{bound}_{jump_rate}.feather")
            results_df_asian_cutoff.to_feather(
                f"MLMC_Results/Results/results_asian_cutoff_{bound}_{jump_rate}.feather")

        # Barrier option
        costs_barrier_fixed_grid.append(costs_barrier_fixed_grid_temp)
        prices_barrier_fixed.append(prices_barrier_fixed_temp)

        costs_barrier_jump_adapted.append(costs_barrier_jump_adapted_temp)
        prices_barrier_adapted.append(prices_barrier_adapted_temp)

        costs_barrier_adapted_cutoff.append(costs_barrier_cutoff_temp)
        prices_barrier_cutoff.append(prices_barrier_cutoff_temp)

        # Asian option
        costs_asian_fixed_grid.append(costs_asian_fixed_grid_temp)
        prices_asian_fixed.append(prices_asian_fixed_temp)

        costs_asian_jump_adapted.append(costs_asian_jump_adapted_temp)
        prices_asian_adapted.append(prices_asian_adapted_temp)

        costs_asian_adapted_cutoff.append(costs_asian_cutoff_temp)
        prices_asian_cutoff.append(prices_asian_cutoff_temp)

    # Barrier Option
    costs_barrier_fixed_grid = np.array(costs_barrier_fixed_grid)
    costs_barrier_jump_adapted = np.array(costs_barrier_jump_adapted)
    costs_barrier_adapted_cutoff = np.array(costs_barrier_adapted_cutoff)

    # Asian Option
    costs_asian_fixed_grid = np.array(costs_asian_fixed_grid)
    costs_asian_jump_adapted = np.array(costs_asian_jump_adapted)
    costs_asian_adapted_cutoff = np.array(costs_asian_adapted_cutoff)

    # Save cost results
    cost_df_barrier_fixed = pd.DataFrame({
        bound: costs_barrier_fixed_grid[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    cost_df_barrier_adapted = pd.DataFrame({
        bound: costs_barrier_jump_adapted[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    cost_df_barrier_cutoff = pd.DataFrame({
        bound: costs_barrier_adapted_cutoff[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)

    cost_df_asian_fixed = pd.DataFrame({
        bound: costs_asian_fixed_grid[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    cost_df_asian_adapted = pd.DataFrame({
        bound: costs_asian_jump_adapted[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    cost_df_asian_cutoff = pd.DataFrame({
        bound: costs_asian_adapted_cutoff[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)

    cost_df_barrier_fixed.to_feather("MLMC_Results/Costs/cost_df_barrier_fixed.feather")
    cost_df_barrier_adapted.to_feather("MLMC_Results/Costs/cost_df_barrier_adapted.feather")
    cost_df_barrier_cutoff.to_feather("MLMC_Results/Costs/cost_df_barrier_cutoff.feather")

    cost_df_asian_fixed.to_feather("MLMC_Results/Costs/cost_df_asian_fixed.feather")
    cost_df_asian_adapted.to_feather("MLMC_Results/Costs/cost_df_asian_adapted.feather")
    cost_df_asian_cutoff.to_feather("MLMC_Results/Costs/cost_df_asian_cutoff.feather")

    # Save prices
    prices_df_barrier_fixed = pd.DataFrame({
        bound: prices_barrier_fixed[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    prices_df_barrier_adapted = pd.DataFrame({
        bound: prices_barrier_adapted[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    prices_df_barrier_cutoff = pd.DataFrame({
        bound: prices_barrier_cutoff[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)

    prices_df_asian_fixed = pd.DataFrame({
        bound: prices_asian_fixed[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    prices_df_asian_adapted = pd.DataFrame({
        bound: prices_asian_adapted[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)
    prices_df_asian_cutoff = pd.DataFrame({
        bound: prices_asian_cutoff[i] for i, bound in
        enumerate(error_bounds)}, index=jump_rates)

    prices_df_barrier_fixed.to_feather("MLMC_Results/Prices/prices_df_barrier_fixed_half.feather")
    prices_df_barrier_adapted.to_feather("MLMC_Results/Prices/prices_df_barrier_adapted.feather")
    prices_df_barrier_cutoff.to_feather("MLMC_Results/Prices/prices_df_barrier_cutoff.feather")

    prices_df_asian_fixed.to_feather("MLMC_Results/Prices/prices_df_asian_fixed_half.feather")
    prices_df_asian_adapted.to_feather("MLMC_Results/Prices/prices_df_asian_adapted.feather")
    prices_df_asian_cutoff.to_feather("MLMC_Results/Prices/prices_df_asian_cutoff.feather")


if __name__ == "__main__":
    main()