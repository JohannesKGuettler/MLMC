"""
Tools perform parallelized Monte Carlo simulations for options.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import numpy as np
import pandas as pd
import discretization as disc
import merton_jump_model as jm
import options as op
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
    return np.sqrt(((jumps_mean ** 2 + jumps_std ** 2) * jump_rate) / y)


def mc_chunk_barrier(fixed, N, M, jm_args, option_args, cutoff_value):
    """
    This function performs a Monte Carlo simulation for a barrier
    option for a given chunk size.
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
        cutoff_value: This value specifies the cutoff value above
        which jumps are simulated.

    Returns:
        A float that specifies the mean price for the given chunk size.
    """
    # Initialize jump model, option and discretization scheme
    local_jm_args = jm_args.copy()
    drift = calculate_drift(local_jm_args["r"], local_jm_args["jump_rate"],
                            local_jm_args["jumps_mean"], local_jm_args["jumps_std"],
                            cutoff_value)
    local_jm_args["drift_coarse"] = drift
    local_jm_args["drift_fine"] = drift
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    option = op.KnockOutOption(**local_option_args)

    payoff_sum = 0

    # Simulate paths and calculate prices
    for i in range(N):
        if i % 1000 == 0:
            print(f"Jump Rate: {option.jump_model.jump_rate}, Option: {type(option)}, Sample: {i}")

        if fixed:
            simulated_path = disc.milstein_fixed_grid_barrier(option.maturity, M, option.S_0, jump_model, option)
        else:
            simulated_path = disc.milstein_jump_adapted_barrier(option.maturity, M, option.S_0,
                                                                jump_model, option,
                                                                cutoff_value, cutoff_value)

        payoff_sum += option.payoff_fine(simulated_path)

    return payoff_sum/N

def mc_chunk_asian(fixed, N, M, jm_args, option_args, cutoff_value):
    """
    This function performs a Monte Carlo simulation for an asian
    option for a given chunk size.
    Args:
        fixed: This boolean specifies if a fixed grid discretization
        scheme is used or a jump adapted one.
        N: This integer specifies the chunk size.
        M: This integer specifies the number of simulated steps in
        the interval.
        jm_args: This dict contains the necessary parameters to
        initialize the jump model.
        option_args: This dict contains the necessary parameters
        to initialize the asian option.
        cutoff_value: This value specifies the cutoff value above
        which jumps are simulated.

    Returns:
        A float that specifies the mean price for the given chunk size.
    """
    local_jm_args = jm_args.copy()
    drift = calculate_drift(local_jm_args["r"], local_jm_args["jump_rate"],
                            local_jm_args["jumps_mean"], local_jm_args["jumps_std"],
                            cutoff_value)
    local_jm_args["drift_coarse"] = drift
    local_jm_args["drift_fine"] = drift
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    option = op.AsianOption(**local_option_args)

    payoff_sum = 0

    for i in range(N):
        if i % 1000 == 0:
            print(f"Jump Rate: {option.jump_model.jump_rate}, Option: {type(option)}, Sample: {i}")
        if fixed:
            simulated_path = disc.milstein_fixed_grid(option.maturity, M, option.S_0, jump_model)
        else:
            simulated_path = disc.milstein_jump_adapted(option.maturity, M, option.S_0,
                                                        jump_model, cutoff_value, cutoff_value)

        payoff_sum += option.payoff_fine(simulated_path)

    return payoff_sum/N

def mc_main():
    # Define option parameters ptions
    r = 0.1
    vola = 0.6
    jumps_mean = 0.1
    jumps_std = 0.2

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

    jump_rates = np.array([1, 5, 10, 15, 20])

    barrier_results_fixed = []
    barrier_results_adapted = []
    barrier_results_cutoff = []

    asian_results_fixed = []
    asian_results_adapted = []
    asian_results_cutoff = []

    N = 1e6
    M = 1e3

    # Define parallelization parameters
    workers = 16
    chunk_sizes = [int(N // workers)] * workers
    chunk_sizes[-1] += N % workers
    chunk_sizes[-1] = int(chunk_sizes[-1])

    for i, jump_rate in enumerate(jump_rates):
        # Define jump model parameters
        cutoff_val = g_inverse(M, jump_rate, jumps_mean, jumps_std)

        args_merton = {"r": r,
                       "jump_rate": jump_rate,
                       "vola": vola,
                       "jumps_mean": jumps_mean,
                       "jumps_std": jumps_std}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Barrier option
            # Fixed grid
            mc_barrier_fixed_results = list(executor.map(mc_chunk_barrier,
                                                         [True] * workers,
                                                         chunk_sizes,
                                                         [int(M)] * workers,
                                                         [args_merton] * workers,
                                                         [args_barrier] * workers,
                                                         [None] * workers,
                                                         chunksize=1))
            barrier_results_fixed.append(np.mean(mc_barrier_fixed_results))

            # Jump Adapted
            mc_barrier_adapted_results = list(executor.map(mc_chunk_barrier,
                                                          [False] * workers,
                                                          chunk_sizes,
                                                          [int(M)] * workers,
                                                          [args_merton] * workers,
                                                          [args_barrier] * workers,
                                                          [None] * workers,
                                                          chunksize=1))
            barrier_results_adapted.append(np.mean(mc_barrier_adapted_results))

            # Cutoff
            mc_barrier_cutoff_results = list(executor.map(mc_chunk_barrier,
                                                           [False] * workers,
                                                           chunk_sizes,
                                                           [int(M)] * workers,
                                                           [args_merton] * workers,
                                                           [args_barrier] * workers,
                                                           [cutoff_val] * workers,
                                                           chunksize=1))
            barrier_results_cutoff.append(np.mean(mc_barrier_cutoff_results))

            # Asian option
            # Fixed grid
            mc_asian_fixed_results = list(executor.map(mc_chunk_asian,
                                                       [True] * workers,
                                                       chunk_sizes,
                                                       [int(M)] * workers,
                                                       [args_merton] * workers,
                                                       [args_asian] * workers,
                                                       [None] * workers,
                                                       chunksize=1))
            asian_results_fixed.append(np.mean(mc_asian_fixed_results))

            # Jump Adapted
            mc_asian_adapted_results = list(executor.map(mc_chunk_asian,
                                                        [False] * workers,
                                                        chunk_sizes,
                                                        [int(M)] * workers,
                                                        [args_merton] * workers,
                                                        [args_asian] * workers,
                                                        [cutoff_val] * workers,
                                                        chunksize=1))
            asian_results_adapted.append(np.mean(mc_asian_adapted_results))

            # Cutoff
            mc_asian_cutoff_results = list(executor.map(mc_chunk_asian,
                                                         [False] * workers,
                                                         chunk_sizes,
                                                         [int(M)] * workers,
                                                         [args_merton] * workers,
                                                         [args_asian] * workers,
                                                         [cutoff_val] * workers,
                                                         chunksize=1))
            asian_results_cutoff.append(np.mean(mc_asian_cutoff_results))

    # Save results
    mc_results_df = pd.DataFrame({"Barrier Fixed Grid": barrier_results_fixed,
                                  "Barrier Jump Adapted": barrier_results_adapted,
                                  "Barrier Cutoff": barrier_results_cutoff,
                                  "Asian Fixed Grid": asian_results_fixed,
                                  "Asian Jump Adapted": asian_results_adapted,
                                  "Asian Cutoff": asian_results_cutoff}
                                  , index=jump_rates)

    mc_results_df.to_feather("MLMC_Results/Prices/mc_benchmark_results.feather")

if __name__ == "__main__":
    mc_main()