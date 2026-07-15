"""
Tools to perform parallelized Multilevel Monte Carlo simulations for options.
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


def mlmc_chunk_0(chunk_size, jm_args, drift, option_args, option_identifier, fixed_grid, cutoff_bound):
    """
    This function performs a Monte Carlo simulation for a barrier
    option for a given chunk size.
    Args:
        chunk_size: This integer specifies the chunk size.
        jm_args: This dict contains the necessary parameters to
        initialize the jump model.
        drift: This float specifies the drift of the jump model.
        option_args: This dict contains the necessary parameters
        to initialize the option.
        option_identifier: This string specifies if the option is a barrier
        option or not.
        fixed_grid: This boolean specifies if a fixed grid is used.
        cutoff_bound: This float specifies how big jumps (in absolute value)
        must be to be explicitly simulated. If None, all jumps are simulated.

    Returns:
        A list of two lists where the first one contains the payoffs
        and the second one contains the associated costs.
    """
    # Initialize jump model, option and discretization scheme
    local_jm_args = jm_args.copy()
    local_jm_args["drift_coarse"] = drift
    local_jm_args["drift_fine"] = drift
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    if option_identifier == "Barrier":
        option = op.KnockOutOption(**local_option_args)
    else:
        option = op.AsianOption(**local_option_args)

    # Simulate paths and calculate payoffs
    Y_sum = 0
    Y_square_sum = 0
    costs = 0

    for n in range(chunk_size):
        if fixed_grid:
            if option_identifier == "Barrier":
                payoff, cost_0 = sample_level_0_fixed(option, jump_model,
                                                      disc.milstein_fixed_grid_barrier,
                                                      option_identifier)
            else:
                payoff, cost_0 = sample_level_0_fixed(option, jump_model,
                                                      disc.milstein_fixed_grid,
                                                      option_identifier)
        else:
            if option_identifier == "Barrier":
                payoff, cost_0 = sample_level_0_jump_adapted(option, jump_model,
                                                             disc.milstein_jump_adapted_barrier,
                                                             cutoff_bound, option_identifier)
            else:
                payoff, cost_0 = sample_level_0_jump_adapted(option, jump_model,
                                                             disc.milstein_jump_adapted,
                                                             cutoff_bound, option_identifier)

        Y_sum += payoff
        Y_square_sum += payoff**2
        costs += cost_0

    return [Y_sum, Y_square_sum, costs]

def sample_level_0_fixed(option, jump_model,
                         discretization_scheme, option_identifier):
    """
    This function computes one sample on level 0 for the MLMC algorithm for
    a fixed grid discretization scheme.
    Args:
        option: This class contains the parameters for the priced option.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        discretization_scheme: This function specifies which discretization
        scheme is used.
        option_identifier: This string specifies if the option is a barrier
        option or not.

    Returns:
        A float that specifies the option's payoff at maturity.
        An integer that specifies how many time stamps were computed.
    """
    if option_identifier == "Barrier":
        simulated_path_0 = discretization_scheme(option.maturity, 1, option.S_0,
                                                 jump_model, option)
    else:
        simulated_path_0 = discretization_scheme(option.maturity, 1, option.S_0,
                                                 jump_model)

    payoff = option.payoff_fine(simulated_path_0)
    num_timestamps = simulated_path_0.shape[0] - 1  # Exclude S_0

    return payoff, num_timestamps

def sample_level_0_jump_adapted(option, jump_model,
                                discretization_scheme, cutoff_bound, option_identifier):
    """
    This function computes one sample on level 0 for the MLMC algorithm for
    a jump adapted discretization scheme.
    Args:
        option: This class contains the parameters for the priced option.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        discretization_scheme: This function specifies which discretization
        scheme is used.
        cutoff_bound: This float specifies how big jumps (in absolute value)
        must be to be explicitly simulated. If None, all jumps are simulated.
        option_identifier: This string specifies if the option is a barrier
        option or not.

    Returns:
        A float that specifies the option's payoff at maturity.
        An integer that specifies how many time stamps were computed.
    """
    if option_identifier == "Barrier":
        simulated_path_0 = discretization_scheme(option.maturity, 1, option.S_0,
                                                 jump_model, option,
                                                 cutoff_bound, cutoff_bound)
    else:
        simulated_path_0 = discretization_scheme(option.maturity, 1, option.S_0,
                                                 jump_model, cutoff_bound, cutoff_bound)

    payoff = option.payoff_fine(simulated_path_0)
    num_timestamps = simulated_path_0.shape[0] - 1  # Exclude S_0

    return payoff, num_timestamps

def mlmc_chunk_l(chunk_size, l, jm_args, drifts, option_args,
                 option_identifier, fixed_grid, cutoff_bounds):
    """
    This function performs a Monte Carlo simulation for a barrier
    option for a given chunk size.
    Args:
        chunk_size: This integer specifies the chunk size.
        l: This integer specifies the level at which a sample is computed.
        discretization steps grows from level to level.
        jm_args: This dict contains the necessary parameters to
        initialize the jump model.
        drifts: This list contains the drifts for the coarse and fine path
        option_args: This dict contains the necessary parameters
        to initialize the option.
        option_identifier: This integer specifies if the option is a barrier 
        option or not.
        fixed_grid: This boolean specifies if a fixed grid is used.
        cutoff_bounds: This list contains the cutoff values above which
        jumps are simulated.

    Returns:
        A list of two lists where the first one contains the payoffs
        and the second one contains the associated costs.
    """
    # Initialize jump model, option and discretization scheme
    local_jm_args = jm_args.copy()
    local_jm_args["drift_coarse"] = drifts[0]
    local_jm_args["drift_fine"] = drifts[1]
    del local_jm_args["r"]
    jump_model = jm.MertonJD(**local_jm_args)

    local_option_args = option_args.copy()
    local_option_args["jump_model"] = jump_model

    if option_identifier == "Barrier":
        option = op.KnockOutOption(**local_option_args)
    else:
        option = op.AsianOption(**local_option_args)

    # Simulate paths and calculate payoffs
    Y_sum = 0
    Y_square_sum = 0
    costs = 0

    for n in range(chunk_size):
        if fixed_grid:
            if option_identifier == "Barrier":
                payoff, cost_l = sample_level_l_fixed(option, jump_model, disc.milstein_fixed_grid_barrier,
                                                      l, option_identifier)
            else:
                payoff, cost_l = sample_level_l_fixed(option, jump_model, disc.milstein_fixed_grid,
                                                      l, option_identifier)
        else:
            if option_identifier == "Barrier":
                payoff, cost_l = sample_level_l_jump_adapted(option, jump_model,
                                                             disc.milstein_jump_adapted_barrier,
                                                             l, cutoff_bounds, option_identifier)
            else:
                payoff, cost_l = sample_level_l_jump_adapted(option, jump_model,
                                                             disc.milstein_jump_adapted,
                                                             l, cutoff_bounds, option_identifier)

        Y_sum += payoff
        Y_square_sum += payoff**2
        costs += cost_l

    return [Y_sum, Y_square_sum, costs]
    

def sample_level_l_fixed(option, jump_model,
                         discretization_scheme, l, option_identifier):
    """
    This function computes one sample on level l > 0 for the MLMC algorithm for
    a fixed grid discretization scheme.
    Args:
        option: This class contains the parameters for the priced option.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        discretization_scheme: This function specifies which discretization
        scheme is used.
        l: This integer specifies the level at which a sample is computed.
        discretization steps grows from level to level.
        option_identifier: This string specifies if the option is a barrier
        option or not.

    Returns:
        A float that specifies the option's payoff at maturity.
        An integer that specifies how many time stamps were computed.
    """
    if option_identifier == "Barrier":
        simulated_path_l = discretization_scheme(option.maturity, 2 ** l,
                                                 option.S_0, jump_model, option)
    else:
        simulated_path_l = discretization_scheme(option.maturity, 2 ** l,
                                                 option.S_0, jump_model)

    payoff_l_1 = option.payoff_fine(simulated_path_l)
    payoff_l_0 = option.payoff_coarse(simulated_path_l)

    num_timestamps = simulated_path_l.shape[0] - 1  # Exclude S_0

    return payoff_l_1 - payoff_l_0, num_timestamps

def sample_level_l_jump_adapted(option, jump_model,
                                discretization_scheme, l, cutoff_bounds,
                                option_identifier):
    """
    This function computes one sample on level l > 0 for the MLMC algorithm for a
    jump adapted discretization scheme.
    Args:
        option: This class contains the parameters for the priced option.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        discretization_scheme: This function specifies which discretization
        scheme is used.
        l: This integer specifies the level at which a sample is computed.
        discretization steps grows from level to level.
        cutoff_bounds: This list contains the cutoff values above which
        jumps are simulated.
        option_identifier: This string specifies if the option is a barrier
        option or not.

    Returns:
        A float that specifies the option's payoff at maturity.
        An integer that specifies how many time stamps were computed.
    """
    if option_identifier == "Barrier":
        simulated_path_l = discretization_scheme(option.maturity, 2 ** l,
                                                 option.S_0, jump_model, option,
                                                 cutoff_bounds[0], cutoff_bounds[1])
    else:
        simulated_path_l = discretization_scheme(option.maturity, 2 ** l,
                                                 option.S_0, jump_model,
                                                 cutoff_bounds[0], cutoff_bounds[1])

    payoff_l_1 = option.payoff_fine(simulated_path_l)
    payoff_l_0 = option.payoff_coarse(simulated_path_l)

    num_timestamps = simulated_path_l.shape[0] - 1  # Exclude S_0

    return payoff_l_1 - payoff_l_0, num_timestamps


def mlmc_main(N_0, L, rates, error_bound,
              jm_args, option_args, option_identifier,
              fixed_grid, cutoff_function, option_name):
        """
        This function runs a geometric multilevel Monte Carlo simulation to price
        an option.
        Args:
             N_0: This array specifies the number of initial samples for the initial
             number of levels.
             L: This integer specifies the initial number of levels.
             rates: This dict contains values for the MLMC rates alpha, beta and gamma
             as well as a boolean for each rate if it is generally known or has
             to be estimated.
             error_bound: This float specifies the maximum error below which
             the algorithm stops.
             jm_args: This dict contains the necessary parameters to
             initialize the jump model.
             option_args: This dict contains the necessary parameters
             to initialize the option.
             option_identifier: This integer specifies if the option is a barrier
             option or not.
             fixed_grid: This boolean specifies if a fixed grid is used.
             cutoff_function: This function is used determine a bound below which
             jumps on a specific level are not simulated. If None, all jumps
             are simulated.

        Returns:
            Dataframe with information on V_l, Mean_Error_l and N_l where l is the index.
            Total costs.
            Option price.
        """
        workers = 16  # Number of threads used

        errors = [np.inf]  # inf is just a placeholder value and isn't used for calculation
        N_0 = np.array(N_0)
        N_l_all = N_0
        samples_needed = N_l_all

        sum_Y = np.zeros(L + 1)
        sum_Y_squared = np.zeros(L + 1)
        num_samples_computed = np.zeros(L + 1, dtype=int)
        C_l_all = np.zeros(L + 1)

        # Define cutoff values
        if cutoff_function is not None:
            h_l_all = [cutoff_function(2**(l + 1),
                                       jm_args["jump_rate"],
                                       jm_args["jumps_mean"],
                                       jm_args["jumps_std"]) for l in range(L+1)]
            drift_l_all = [calculate_drift(jm_args["r"],
                                           jm_args["jump_rate"],
                                           jm_args["jumps_mean"],
                                           jm_args["jumps_std"],
                                           h_l_all[l]) for l in range(L + 1)]

        else:
            h_l_all = [None] * (L + 1)
            drift_l_all = [calculate_drift(jm_args["r"],
                                           jm_args["jump_rate"],
                                           jm_args["jumps_mean"],
                                           jm_args["jumps_std"], None)] * (L + 1)

        while np.any(samples_needed > 0):
            print("--------------------")
            print(f"{option_name}, error: {error_bound}, jump rate: {jm_args["jump_rate"]}")
            print(f"error: {np.max(errors)}, needed: {error_bound / np.sqrt(2)}")
            print(f"N_l: {N_l_all}")
            print(f"Samples computed: {num_samples_computed}")
            print(f"Samples needed: {samples_needed}")
            print("--------------------")
            # Estimate payoff for l=0
            amount_new_samples = int(samples_needed[0])
            if amount_new_samples > 0:
                num_samples_computed[0] += amount_new_samples

                chunk_sizes_0 = [int(amount_new_samples // workers)] * workers
                chunk_sizes_0[-1] += amount_new_samples % workers
                chunk_sizes_0[-1] = int(chunk_sizes_0[-1])

                with ProcessPoolExecutor(max_workers=workers) as ex:
                    return_values = list(ex.map(mlmc_chunk_0,
                                                chunk_sizes_0,
                                                [jm_args]*workers,
                                                [drift_l_all[0]]*workers,
                                                [option_args]*workers,
                                                [option_identifier]*workers,
                                                [fixed_grid]*workers,
                                                [h_l_all[0]]*workers,
                                                chunksize=1))

                # Update result arrays
                for worker_res in return_values:
                    sum_Y[0] += worker_res[0]
                    sum_Y_squared[0] += worker_res[1]
                    C_l_all[0] += worker_res[2]

            # Estimate payoffs for l>1
            for l in range(1, L+1):
                amount_new_samples = int(samples_needed[l])

                # Check if new samples on current level are needed
                if amount_new_samples > 0:
                    num_samples_computed[l] += amount_new_samples

                    chunk_sizes_l = [int(amount_new_samples // workers)] * workers
                    chunk_sizes_l[-1] += amount_new_samples % workers
                    chunk_sizes_l[-1] = int(chunk_sizes_l[-1])

                    with ProcessPoolExecutor(max_workers=workers) as ex:
                        return_values = list(ex.map(mlmc_chunk_l,
                                                    chunk_sizes_l,
                                                    [l]*workers,
                                                    [jm_args] * workers,
                                                    [[drift_l_all[l-1], drift_l_all[l]]] * workers,
                                                    [option_args] * workers,
                                                    [option_identifier] * workers,
                                                    [fixed_grid] * workers,
                                                    [[h_l_all[l-1], h_l_all[l]]] * workers,
                                                    chunksize=1))

                    # Update result arrays
                    for worker_res in return_values:
                        sum_Y[l] += worker_res[0]
                        sum_Y_squared[l] += worker_res[1]
                        C_l_all[l] += worker_res[2]

            Y_hat = np.abs(sum_Y / num_samples_computed)
            V_l_all = np.maximum(sum_Y_squared/(N_l_all - 1) - (sum_Y/(N_l_all - 1))**2, 0)
            C_l_all_mean = C_l_all / N_l_all

            # Avoid zero values for V_l and Y_hat for new levels
            # (can happen when there are few samples)
            for l in range(2, L+1):
                Y_hat[l] = max(Y_hat[l], 0.5 * Y_hat[l-1] / 2**rates["alpha"][0])
                V_l_all[l] = max(V_l_all[l], 0.5 * V_l_all[l-1] / 2**rates["beta"][0])

            # Estimate alpha, beta and theta using regression if not given
            levels = np.arange(1, L + 1)
            # Alpha
            if not rates["alpha"][1]:
                # Filter out zero values to avoid -inf values (should only occur on level 1
                # but better safe than sorry)
                Y_hat_mask = Y_hat[1:] > 0
                if np.count_nonzero(Y_hat_mask) > 1:
                    A_error = np.column_stack((levels[Y_hat_mask], np.ones(np.count_nonzero(Y_hat_mask))))
                    y_error = np.log2(Y_hat[1:][Y_hat_mask])
                    x_variance, *_ = np.linalg.lstsq(A_error, y_error)
                    alpha_new = max(0.5, -x_variance[0])
                else:
                    alpha_new = 0.5

                rates["alpha"][0] = alpha_new

            # Beta
            if not rates["beta"][1]:
                # Filter out zero values to avoid -inf values (should only occur on level 1
                # but better safe than sorry)
                V_l_mask = V_l_all[1:] > 0
                if np.count_nonzero(V_l_mask) > 1:
                    A_variance = np.column_stack((levels[V_l_mask], np.ones(np.count_nonzero(V_l_mask))))
                    y_variance = np.log2(V_l_all[1:][V_l_mask])
                    x_variance, *_ = np.linalg.lstsq(A_variance, y_variance)
                    beta_new = max(0.5, -x_variance[0])
                else:
                    beta_new = 0.5

                rates["beta"][0] = beta_new

            # Theta
            if not rates["theta"][1]:
                A = np.column_stack((levels, np.ones(L)))
                y_costs = np.log2(C_l_all_mean[1:])
                x_costs, *_ = np.linalg.lstsq(A, y_costs)
                theta_new = max(0.5, x_costs[0])
                rates["theta"][0] = theta_new

            # Update sample sizes
            N_l_all = np.ceil(
                2 / (error_bound ** 2) * np.sqrt(V_l_all / C_l_all_mean) * np.sum(np.sqrt(V_l_all * C_l_all_mean)))
            samples_needed = np.maximum(N_l_all - num_samples_computed, 0)

            # If almost all needed samples calculated check convergence
            if np.all(samples_needed <= 0.01*N_l_all):

                # Check convergence
                Y_hat_last_three = Y_hat[-3:]
                errors = Y_hat_last_three / (2 ** rates["alpha"][0] - 1)
                convergence_achieved = True if np.max(errors) < error_bound / np.sqrt(2) else False

                if not convergence_achieved:
                    L += 1

                    # Expand parameter arrays
                    sum_Y = np.append(sum_Y, [0])
                    sum_Y_squared = np.append(sum_Y_squared, [0])
                    num_samples_computed = np.append(num_samples_computed, [0])

                    V_l_all = np.append(V_l_all, [V_l_all[-1] / 2**rates["beta"][0]])
                    C_l_all = np.append(C_l_all, [0])
                    C_l_all_mean = np.append(C_l_all_mean, [C_l_all_mean[-1] *
                                                            2 ** rates["theta"][0]])

                    if cutoff_function is not None:
                        h_l_all.append(cutoff_function(2**(L + 1),
                                                       jm_args["jump_rate"],
                                                       jm_args["jumps_mean"],
                                                       jm_args["jumps_std"]))
                        drift_l_all.append(calculate_drift(jm_args["r"],
                                                           jm_args["jump_rate"],
                                                           jm_args["jumps_mean"],
                                                           jm_args["jumps_std"],
                                                           h_l_all[-1]))

                    else:
                        h_l_all.append(None)
                        drift_l_all.append(drift_l_all[-1])

                    N_l_all = np.ceil(
                        2 / (error_bound ** 2) * np.sqrt(V_l_all / C_l_all_mean) *
                        np.sum(np.sqrt(V_l_all * C_l_all_mean)))

                    samples_needed = np.maximum(N_l_all - num_samples_computed, 0)

        price = np.sum(sum_Y / num_samples_computed)

        return_df = pd.DataFrame({
                        "V_l": V_l_all,
                        "Mean_Error_l": Y_hat,
                        "N_l": N_l_all,
                        "C_l": C_l_all_mean})

        return return_df, np.sum(C_l_all), price
