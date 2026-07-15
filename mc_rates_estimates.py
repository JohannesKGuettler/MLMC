"""
Tools to get estimates for MLMC convergence rate of error and sample variance.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import pandas as pd
import numpy as np
import options as op
import merton_jump_model as jm
import discretization as disc
from concurrent.futures import ProcessPoolExecutor
from scipy import stats


def calculate_drift(r, jump_rate, jumps_mean, jumps_std, h_l):
    """
    This function calculates the drift for the merton jump model.
    Args:
        r: A float that specifies the risk-free interest rate.
        jump_rate: A float that specifies the jump rate.
        jumps_mean: A float that specifies the mean jump size.
        jumps_std: A float that specifies the standard deviation of the jump size.
        h_l: A float that specifies the jump cutoff value on level l. It is needed to correct
        the drift. If None, the drift is constant for each level.

    Returns:
        A float that specifies the drift for the merton jump model.
    """
    if h_l is None:
        drift = r - jump_rate * (np.exp(jumps_mean + jumps_std**2 / 2) - 1)
    else:
        drift = r - jump_rate * (
                np.exp(jumps_mean + jumps_std**2 / 2) * (
                    stats.norm.cdf((jumps_mean + jumps_std**2 - h_l)/jumps_std)
                    + stats.norm.cdf((-h_l - (jumps_mean + jumps_std**2))/jumps_std))
                - stats.norm.cdf((jumps_mean - h_l)/jumps_std)
                - stats.norm.cdf((-h_l - jumps_mean)/jumps_std))

    return drift


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
    return np.sqrt(((jumps_mean**2 + jumps_std**2) * jump_rate) / y)


def monte_carlo_alpha_beta_chunk_barrier(fixed, N, M, jm_args, option_args, cutoff_values=[None, None]):
    """
    This method runs a standard Monte Carlo simulation to estimate V_l and
    |E[P_l - P_{l-1}]| for a barrier option.
    Args:
        fixed: This boolean specifies if a fixed grid discretization
        scheme is used or a jump adapted one.
        N: This integer specifies the chunk size.
        M: This integer specifies the number of simulated steps in
        the interval.
        jm_args: This dict contains the necessary parameters to
        initialize the jump model.
        option_args: This dict contains the necessary parameters
        to initialize the barrier option.
        cutoff_values: This list contains the jump cutoff values for
        the coarse and fine grid.

    Returns:
        Sum of P_l - P_{l-1}
        Sum of (P_l - P_{l-1})^2
    """
    # Initialize jump model, option and discretization scheme
    local_jm_args = jm_args.copy()
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    option = op.KnockOutOption(**local_option_args)

    sum_Y = 0
    sum_Y_squared = 0
    for i in range(N):
        if i % 1000 == 0:
            print(f"Jump Rate: {option.jump_model.jump_rate}, Option: {type(option)}, Sample: {i}")
        if fixed:
            simulated_path = disc.milstein_fixed_grid_barrier(option.maturity, M, option.S_0,
                                                              jump_model, option)
        else:
            simulated_path = disc.milstein_jump_adapted_barrier(option.maturity, M, option.S_0,
                                                                jump_model, option,
                                                                cutoff_values[0], cutoff_values[1])

        Y = option.payoff_fine(simulated_path) - option.payoff_coarse(simulated_path)
        sum_Y += Y
        sum_Y_squared += Y**2

    return [sum_Y, sum_Y_squared]


def monte_carlo_alpha_beta_chunk_asian(fixed, N, M, jm_args, option_args, cutoff_values=[None, None]):
    """
    This method runs a standard Monte Carlo simulation to estimate V_l and
    |E[P_l - P_{l-1}]| for an asian option.
    Args:
        fixed: This boolean specifies if a fixed grid discretization
        scheme is used or a jump adapted one.
        N: This integer specifies the chunk size.
        M: This integer specifies the number of simulated steps in
        the interval.
        jm_args: This dict contains the necessary parameters to
        initialize the jump model.
        option_args: This dict contains the necessary parameters
        to initialize the barrier option.
        cutoff_values: This list contains the jump cutoff values for
        the coarse and fine grid.

    Returns:
        Sum of P_l - P_{l-1}
        Sum of (P_l - P_{l-1})^2
    """
    # Initialize jump model, option and discretization scheme
    local_jm_args = jm_args.copy()
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    option = op.AsianOption(**local_option_args)

    sum_Y = 0
    sum_Y_squared = 0
    for i in range(N):
        if i % 1000 == 0:
            print(f"Jump Rate: {option.jump_model.jump_rate}, Option: {type(option)}, Sample: {i}")
        if fixed:
            simulated_path = disc.milstein_fixed_grid(option.maturity, M, option.S_0, jump_model)
        else:
            simulated_path = disc.milstein_jump_adapted(option.maturity, M, option.S_0, jump_model,
                                                              cutoff_values[0], cutoff_values[1])

        Y = option.payoff_fine(simulated_path) - option.payoff_coarse(simulated_path)
        sum_Y += Y
        sum_Y_squared += Y**2

    return [sum_Y, sum_Y_squared]


def estimate_rates(df):
    """
    This function estimates the convergence rates for the error and sample variance
    for a given option using least squares.
    Args:
        df: This dataframe contains the error and sample variance for each level.

    Returns:
        A float the specifies the estimated error convergence rate.
        A float that specifies the estimated sample variance convergence rate.
    """
    log_var = np.log2(df["V_l"].to_numpy())
    log_err = np.log2(np.abs(df["Mean_Error_l"].to_numpy()))

    # Start regression after the maximum of each quantity
    start_var = np.argmax(log_var)
    start_err = np.argmax(log_err)

    x_var = df.index.to_numpy()[start_var:]
    x_err = df.index.to_numpy()[start_err:]

    # Create system matrix
    A_var = np.column_stack((x_var, np.ones_like(x_var)))
    A_err = np.column_stack((x_err, np.ones_like(x_err)))

    slope_var, _ = np.linalg.lstsq(A_var, log_var[start_var:])[0]
    slope_err, _ = np.linalg.lstsq(A_err, log_err[start_err:])[0]

    return -slope_err, -slope_var


def get_convergence_rates(jump_rate):
    """
    This function estimates the weak convergence rates and the
    V_l convergence rates for a fixed grid and a jump adapted
    discretization scheme in combination with a barrier and an
    asian option. It uses a MC simulation to estimate the required
    expected values and variances.
    Args:
        jump_rate: A float that specifies the jump rate.

    Returns: A list for each option and grid type that contains the
    error and sample variance convergence rates.
    """
    # Define parameters
    # Jump models
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

    max_level = 10
    num_samples = int(1e6)

    # Define parallelization parameters
    workers = 16
    chunk_sizes = [int(num_samples // workers)] * workers
    chunk_sizes[-1] += num_samples % workers
    chunk_sizes[-1] = int(chunk_sizes[-1])

    variances_barrier_fixed = []
    variances_barrier_adapted = []
    variances_barrier_cutoff = []
    variances_asian_fixed = []
    variances_asian_adapted = []
    variances_asian_cutoff = []

    means_barrier_fixed = []
    means_barrier_adapted = []
    means_barrier_cutoff = []
    means_asian_fixed = []
    means_asian_adapted = []
    means_asian_cutoff = []

    levels = [l for l in range(1, max_level + 1)]
    cutoff_values = [g_inverse(2**(l+1), jump_rate, jumps_mean, jumps_std) for l in range(max_level + 1)]

    for level in levels:
        drift_coarse = calculate_drift(r, jump_rate, jumps_mean, jumps_std, cutoff_values[level-1])
        drift_fine = calculate_drift(r, jump_rate, jumps_mean, jumps_std, cutoff_values[level])

        args_merton = {"r": r,
                       "drift_coarse": drift_coarse,
                       "drift_fine": drift_fine,
                       "vola": vola,
                       "jump_rate": jump_rate,
                       "jumps_mean": jumps_mean,
                       "jumps_std": jumps_std}

        Y_sum_barrier_fixed = 0
        Y_sum_barrier_adapted = 0
        Y_sum_barrier_cutoff = 0
        Y_sum_asian_fixed = 0
        Y_sum_asian_adapted = 0
        Y_sum_asian_cutoff = 0

        Y_squared_sum_barrier_fixed = 0
        Y_squared_sum_barrier_adapted = 0
        Y_squared_sum_barrier_cutoff = 0
        Y_squared_sum_asian_fixed = 0
        Y_squared_sum_asian_adapted = 0
        Y_squared_sum_asian_cutoff = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Barrier Option
            # Fixed grid
            print("##################################")
            print(f"barrier fixed grid, level: {level}")
            barrier_fixed_results = list(executor.map(monte_carlo_alpha_beta_chunk_barrier,
                                                      [True] * workers,
                                                      chunk_sizes,
                                                      [int(2**level)] * workers,
                                                      [args_merton] * workers,
                                                      [args_barrier] * workers,
                                                      chunksize=1))

            # Jump adapted
            print("##################################")
            print(f"barrier jump adapted, level: {level}")
            barrier_adapted_results = list(executor.map(monte_carlo_alpha_beta_chunk_barrier,
                                                       [False] * workers,
                                                       chunk_sizes,
                                                       [int(2 ** level)] * workers,
                                                       [args_merton] * workers,
                                                       [args_barrier] * workers,
                                                       [[None, None]] * workers,
                                                       chunksize=1))

            # Cutoff
            print("##################################")
            print(f"barrier cutoff, level: {level}")
            barrier_cutoff_results = list(executor.map(monte_carlo_alpha_beta_chunk_barrier,
                                                       [False] * workers,
                                                       chunk_sizes,
                                                       [int(2 ** level)] * workers,
                                                       [args_merton] * workers,
                                                       [args_barrier] * workers,
                                                       [[cutoff_values[level - 1], cutoff_values[level]]] * workers,
                                                       chunksize=1))

            # Asian Option
            # Fixed grid
            print("##################################")
            print(f"asian fixed grid, level: {level}")
            asian_fixed_results = list(executor.map(monte_carlo_alpha_beta_chunk_asian,
                                                    [True] * workers,
                                                    chunk_sizes,
                                                    [int(2 ** level)] * workers,
                                                    [args_merton] * workers,
                                                    [args_asian] * workers,
                                                    chunksize=1))

            # Jump adapted
            print("##################################")
            print(f"asian jump adapted, level: {level}")
            asian_adapted_results = list(executor.map(monte_carlo_alpha_beta_chunk_asian,
                                                     [False] * workers,
                                                     chunk_sizes,
                                                     [int(2 ** level)] * workers,
                                                     [args_merton] * workers,
                                                     [args_asian] * workers,
                                                     [[None, None]] * workers,
                                                     chunksize=1))

            # Cutoff
            print("##################################")
            print(f"asian cutoff, level: {level}")
            asian_cutoff_results = list(executor.map(monte_carlo_alpha_beta_chunk_asian,
                                                      [False] * workers,
                                                      chunk_sizes,
                                                      [int(2 ** level)] * workers,
                                                      [args_merton] * workers,
                                                      [args_asian] * workers,
                                                      [[cutoff_values[level - 1], cutoff_values[level]]] * workers,
                                                      chunksize=1))

        # Combine results
        # Barrier
        for worker_res in barrier_fixed_results:
            Y_sum_barrier_fixed += worker_res[0]
            Y_squared_sum_barrier_fixed += worker_res[1]

        variances_barrier_fixed.append(np.maximum(Y_squared_sum_barrier_fixed / (num_samples - 1) \
                                                  - (Y_sum_barrier_fixed / (num_samples - 1)) ** 2, 0))
        means_barrier_fixed.append(np.abs(Y_sum_barrier_fixed / num_samples))

        for worker_res in barrier_adapted_results:
            Y_sum_barrier_adapted += worker_res[0]
            Y_squared_sum_barrier_adapted += worker_res[1]

        variances_barrier_adapted.append(np.maximum(Y_squared_sum_barrier_adapted / (num_samples - 1) \
                                                    - (Y_sum_barrier_adapted / (num_samples - 1)) ** 2, 0))
        means_barrier_adapted.append(np.abs(Y_sum_barrier_adapted / num_samples))

        for worker_res in barrier_cutoff_results:
            Y_sum_barrier_cutoff += worker_res[0]
            Y_squared_sum_barrier_cutoff += worker_res[1]

        variances_barrier_cutoff.append(np.maximum(Y_squared_sum_barrier_cutoff / (num_samples - 1) \
                                                    - (Y_sum_barrier_cutoff / (num_samples - 1)) ** 2, 0))
        means_barrier_cutoff.append(np.abs(Y_sum_barrier_cutoff / num_samples))

        # Asian
        for worker_res in asian_fixed_results:
            Y_sum_asian_fixed += worker_res[0]
            Y_squared_sum_asian_fixed += worker_res[1]

        variances_asian_fixed.append(np.maximum(Y_squared_sum_asian_fixed / (num_samples - 1) \
                                                - (Y_sum_asian_fixed / (num_samples - 1)) ** 2, 0))
        means_asian_fixed.append(np.abs(Y_sum_asian_fixed / num_samples))

        for worker_res in asian_adapted_results:
            Y_sum_asian_adapted += worker_res[0]
            Y_squared_sum_asian_adapted += worker_res[1]

        variances_asian_adapted.append(np.maximum(Y_squared_sum_asian_adapted / (num_samples - 1) \
                                                  - (Y_sum_asian_adapted / (num_samples - 1)) ** 2, 0))
        means_asian_adapted.append(np.abs(Y_sum_asian_adapted / num_samples))

        for worker_res in asian_cutoff_results:
            Y_sum_asian_cutoff += worker_res[0]
            Y_squared_sum_asian_cutoff += worker_res[1]

        variances_asian_cutoff.append(np.maximum(Y_squared_sum_asian_cutoff / (num_samples - 1) \
                                                    - (Y_sum_asian_cutoff / (num_samples - 1)) ** 2, 0))
        means_asian_cutoff.append(np.abs(Y_sum_asian_cutoff / num_samples))

    # Combine results of each level
    # Barrier
    results_df_barrier_fixed = pd.DataFrame({"V_l": variances_barrier_fixed,
                                             "Mean_Error_l": means_barrier_fixed})
    results_df_barrier_fixed.index = levels

    results_df_barrier_adapted = pd.DataFrame({"V_l": variances_barrier_adapted,
                                               "Mean_Error_l": means_barrier_adapted})
    results_df_barrier_adapted.index = levels

    results_df_barrier_cutoff = pd.DataFrame({"V_l": variances_barrier_cutoff,
                                               "Mean_Error_l": means_barrier_cutoff})
    results_df_barrier_cutoff.index = levels

    # Asian
    results_df_asian_fixed = pd.DataFrame({"V_l": variances_asian_fixed,
                                           "Mean_Error_l": means_asian_fixed})
    results_df_asian_fixed.index = levels

    results_df_asian_adapted = pd.DataFrame({"V_l": variances_asian_adapted,
                                             "Mean_Error_l": means_asian_adapted})
    results_df_asian_adapted.index = levels

    results_df_asian_cutoff = pd.DataFrame({"V_l": variances_asian_cutoff,
                                             "Mean_Error_l": means_asian_cutoff})
    results_df_asian_cutoff.index = levels

    # Save results
    results_df_barrier_fixed.to_feather(f"MC_Rates_Estimates_Results/results_MC_barrier_fixed_{jump_rate}_test.feather")
    results_df_barrier_adapted.to_feather(f"MC_Rates_Estimates_Results/results_MC_barrier_adapted_{jump_rate}.feather")
    results_df_barrier_cutoff.to_feather(f"MC_Rates_Estimates_Results/results_MC_barrier_cutoff_{jump_rate}.feather")

    results_df_asian_fixed.to_feather(f"MC_Rates_Estimates_Results/results_MC_asian_fixed_{jump_rate}_test.feather")
    results_df_asian_adapted.to_feather(f"MC_Rates_Estimates_Results/results_MC_asian_adapted_{jump_rate}.feather")
    results_df_asian_cutoff.to_feather(f"MC_Rates_Estimates_Results/results_MC_asian_cutoff_{jump_rate}.feather")

    # Estimate rates via least squares
    # Barrier
    alpha_barrier_fixed, beta_barrier_fixed = estimate_rates(results_df_barrier_fixed)
    alpha_barrier_adapted, beta_barrier_adapted = estimate_rates(results_df_barrier_adapted)
    alpha_barrier_cutoff, beta_barrier_cutoff = estimate_rates(results_df_barrier_cutoff)

    # Asian
    alpha_asian_fixed, beta_asian_fixed = estimate_rates(results_df_asian_fixed)
    alpha_asian_adapted, beta_asian_adapted = estimate_rates(results_df_asian_adapted)
    alpha_asian_cutoff, beta_asian_cutoff = estimate_rates(results_df_asian_cutoff)

    return [alpha_barrier_fixed, beta_barrier_fixed], [alpha_barrier_adapted, beta_barrier_adapted],\
            [alpha_barrier_cutoff, beta_barrier_cutoff], [alpha_asian_fixed, beta_asian_fixed],\
            [alpha_asian_adapted, beta_asian_adapted], [alpha_asian_cutoff, beta_asian_cutoff]

if __name__ == "__main__":
    jump_rates = np.array([1, 5, 10, 15, 20])

    estimates_alpha_df = pd.DataFrame(columns=["Barrier Fixed Grid", "Barrier Jump Adapted", "Barrier Cutoff",
                                               "Asian Fixed Grid", "Asian Jump Adapted", "Asian Cutoff"],
                                      index=jump_rates)
    estimates_beta_df = pd.DataFrame(columns=["Barrier Fixed Grid", "Barrier Jump Adapted", "Barrier Cutoff",
                                              "Asian Fixed Grid", "Asian Jump Adapted", "Asian Cutoff"],
                                     index=jump_rates)

    for jump_rate in jump_rates:
        results_barrier_fixed, results_barrier_adapted, \
            results_barrier_cutoff, results_asian_fixed, \
            results_asian_adapted, results_asian_cutoff = get_convergence_rates(jump_rate)

        estimates_alpha_df.loc[jump_rate, :] = [results_barrier_fixed[0], results_barrier_adapted[0],
                                                results_barrier_cutoff[0], results_asian_fixed[0],
                                                results_asian_adapted[0], results_asian_cutoff[0]]
        estimates_beta_df.loc[jump_rate, :] = [results_barrier_fixed[1], results_barrier_adapted[1],
                                               results_barrier_cutoff[1], results_asian_fixed[1],
                                               results_asian_adapted[1], results_asian_cutoff[1]]

    estimates_alpha_df.to_feather("Rates_Estimates/MC_rates_estimates_alpha_test.feather")
    estimates_beta_df.to_feather("Rates_Estimates/MC_rates_estimates_beta_test.feather")