"""
Tools to create discretizations of jump diffusion processes.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import numpy as np


def milstein_fixed_grid(T, M, S_0, jump_model):
    """
    This function implements the Milstein discretization scheme for a fixed
    time grid.
    Args:
        T: This float specifies the length of the interval.
        M: This integer specifies the number of simulated steps in
        the interval.
        S_0: This float specifies the value of S in t=0.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)

    Returns:
        A 2-D array with a discretization of a jump-diffusion process, a
        coarse version of the simulated jump-diffusion process, and the
        corresponding timestamps.
    """
    delta_t = T / M
    mu_fine = jump_model.drift_fine
    mu_coarse = jump_model.drift_coarse
    std = jump_model.vola

    jump_diffusion_process = [[S_0, S_0, 0]]

    delta_B_coarse = 0
    jump_times_sum = 0
    jumps_occurred = []
    jumps_coarse = []

    for t in range(M):
        delta_B = np.sqrt(delta_t) * np.random.normal()
        delta_B_coarse += delta_B

        # Simulate change of S caused by diffusion process
        S_t_0_fine = jump_diffusion_process[-1][0]
        S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * delta_t + std * S_t_0_fine * \
                     delta_B

        # Simulate change of S caused by Ito correction
        correction_fine = 0.5 * std ** 2 * S_t_0_fine * \
                          (delta_B ** 2 - delta_t)
        S_t_1_fine += correction_fine

        # Simulate change of S caused by jump process
        while jump_times_sum <= delta_t:
            jump_times_sum += jump_model.simulate_jump_time()
            jump_size, _, _ = jump_model.simulate_jump_size()

            jumps_occurred.append(jump_size)
            jumps_coarse.append(jump_size)

        jump_times_sum = jump_times_sum - delta_t
        jumps_occurred_arr = np.array(jumps_occurred[:-1])

        if len(jumps_occurred_arr) > 0:
            jumps_occurred = [jumps_occurred[-1]]

        jump_increments = np.sum(jumps_occurred_arr * S_t_0_fine)
        S_t_1_fine += jump_increments

        # Check if point is contained in coarse path and only
        # simulate change in S (for coarse path) in this case
        if (t+1) % 2 == 0:
            # Simulate change of S caused by diffusion process
            S_t_0_coarse = jump_diffusion_process[-2][1]
            S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * 2 * delta_t + std * S_t_0_coarse * \
                           delta_B_coarse

            # Simulate change of S caused by Ito correction
            correction_coarse = 0.5 * std ** 2 * S_t_0_coarse * \
                              (delta_B_coarse ** 2 - 2*delta_t)
            S_t_1_coarse += correction_coarse

            # Simulate change of S caused by jump process
            jumps_occurred_arr_coarse = np.array(jumps_coarse[:-1])
            jumps_coarse = [jumps_coarse[-1]]
            jump_increments_coarse = np.sum(jumps_occurred_arr_coarse * S_t_0_coarse)
            S_t_1_coarse += jump_increments_coarse

            delta_B_coarse = 0
        else:
            S_t_1_coarse = np.nan

        jump_diffusion_process.append([S_t_1_fine, S_t_1_coarse, (t+1) * delta_t])

    return np.array(jump_diffusion_process)

def milstein_jump_adapted(T, M, S_0, jump_model, 
                          cutoff_l_0, cutoff_l_1):
    """
    This function implements the jump adapted Milstein discretization
    scheme.
    Args:
        T: This float specifies the length of the interval.
        M: This integer specifies the number of simulated steps in
        the interval.
        S_0: This float specifies the value of S in t=0.
        cutoff_l_0: This float specifies how big jumps
        (in absolute value) must be to be explicitly simulated
        on the coarse level. If None, there is no cutoff.
        cutoff_l_1: This float specifies how big jumps
        (in absolute value) must be to be explicitly simulated
        on the fine level. If None, there is no cutoff.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)

    Returns:
        A 2-D array with a discretization of a jump-diffusion process, a
        coarse version of the simulated jump-diffusion process, and the
        corresponding timestamps.
    """
    mu_fine = jump_model.drift_fine
    mu_coarse = jump_model.drift_coarse
    std = jump_model.vola

    # Initialize algorithm
    fixed_grid = np.linspace(0, T, M+1)
    jump_diffusion_process = [[S_0, S_0, S_0, S_0, 0]]

    # Initialize first jump
    delta_jump_fine = jump_model.simulate_jump_time()
    jump_time_sum = delta_jump_fine
    last_simulated_jump_time_fine = 0
    last_simulated_jump_time_coarse = 0

    # Initialize path variables
    S_t_0_fine = jump_diffusion_process[-1][0]
    S_t_0_coarse = jump_diffusion_process[-1][1]
    delta_jump_coarse = delta_jump_fine

    intermediate_brownian_increments = []

    for j in range(1, M+1):
        t_1 = fixed_grid[j]
        t_0 = fixed_grid[j-1]

        # Simulate jumps and S between t_0 and t_1
        while jump_time_sum < t_1:
            jump_size, simulated_fine, simulated_coarse = (
                jump_model.simulate_jump_size(cutoff_l_0, cutoff_l_1))

            if simulated_fine:
                # Simulate change in fine path
                # Drift and diffusion
                delta_B = np.sqrt(delta_jump_fine) * np.random.normal()
                intermediate_brownian_increments.append(delta_B)
                S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * delta_jump_fine + std * S_t_0_fine * delta_B

                # Ito correction
                S_t_1_fine += 0.5 * std ** 2 * S_t_0_fine * (delta_B ** 2 - delta_jump_fine)

                # Jump
                S_t_1_fine_minus = S_t_1_fine
                S_t_1_fine += S_t_1_fine * jump_size

                # Update variables on fine path
                S_t_0_fine = S_t_1_fine
                last_simulated_jump_time_fine = jump_time_sum
                delta_jump = jump_model.simulate_jump_time()
                delta_jump_fine = delta_jump

                if simulated_coarse:
                    # Simulate change in coarse path
                    # Drift and diffusion
                    delta_B_coarse = np.sum(intermediate_brownian_increments)
                    S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * delta_jump_coarse + \
                                   std * S_t_0_coarse * delta_B_coarse

                   # Ito correction
                    S_t_1_coarse += 0.5 * std ** 2 * S_t_0_coarse * (delta_B_coarse ** 2 - delta_jump_coarse)

                    # Jump
                    S_t_1_coarse_minus = S_t_1_coarse
                    S_t_1_coarse += S_t_1_coarse * jump_size

                    # Update variables for coarse path
                    S_t_0_coarse = S_t_1_coarse
                    delta_jump_coarse = delta_jump
                    last_simulated_jump_time_coarse = jump_time_sum
                    intermediate_brownian_increments = []

                else:
                    S_t_1_coarse = np.nan
                    S_t_1_coarse_minus = np.nan
                    delta_jump_coarse += delta_jump

                # Append results and update variables for fine path
                jump_diffusion_process.append([S_t_1_fine, S_t_1_fine_minus,
                                               S_t_1_coarse, S_t_1_coarse_minus, jump_time_sum])

            else:
                delta_jump = jump_model.simulate_jump_time()
                delta_jump_fine += delta_jump
                delta_jump_coarse += delta_jump

            jump_time_sum = delta_jump + jump_time_sum

        # Simulate S between last jump and t_1
        # Drift and diffusion
        dt_fine = t_1 - max(t_0, last_simulated_jump_time_fine)
        delta_B = np.sqrt(dt_fine) * np.random.normal()
        intermediate_brownian_increments.append(delta_B)
        S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * dt_fine + std * S_t_0_fine * delta_B

        # Ito correction
        S_t_1_fine += 0.5 * std ** 2 * S_t_0_fine * (delta_B ** 2 - dt_fine)

        # Check if point is contained in coarse path and only
        # simulate change in S (for coarse path) in this case
        if j % 2 == 0:
            dt_coarse = t_1 - max(fixed_grid[j-2], last_simulated_jump_time_coarse)
            delta_B_coarse = np.sum(intermediate_brownian_increments)
            # Drift and diffusion
            S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * dt_coarse + \
                           std * S_t_0_coarse * delta_B_coarse

            # Ito correction
            S_t_1_coarse += 0.5 * std ** 2 * S_t_0_coarse * (delta_B_coarse ** 2 - dt_coarse)

            # Update variables
            delta_jump_coarse = jump_time_sum - t_1
            intermediate_brownian_increments = []
            S_t_0_coarse = S_t_1_coarse
        else:
            S_t_1_coarse = np.nan

        jump_diffusion_process.append([S_t_1_fine, S_t_1_fine,
                                       S_t_1_coarse, S_t_1_coarse, t_1])

        S_t_0_fine = S_t_1_fine
        delta_jump_fine = jump_time_sum - t_1

    return np.array(jump_diffusion_process)


def evaluate_brownian_interpolant(S_t_0, S_t_1, t_0, t_1,
                                  delta_B_intermediate, delta_B_coarse, t,
                                  jump_model):
    """
    This function evaluates the brownian interpolant at a given time.
    Args:
        S_t_0: This float specifies the stock price at t_0.
        S_t_1: This float specifies the stock price at t_1.
        t_0: This float specifies the time at t_0.
        t_1: This float specifies the time at t_1.
        delta_B_intermediate: This float specifies the brownian increment
        between t_0 and t.
        delta_B_coarse: This float specifies the brownian increment
        between t_0 and t_1. 
        t: This float specifies at what time the interpolant is evaluated.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)

    Returns:
        A float that represents the interpolated value.
    """
    v = (t - t_0) / (t_1 - t_0)
    brownian_part = delta_B_intermediate - v*delta_B_coarse
    price_interpolation = S_t_0 + v*(S_t_1 - S_t_0)

    S_t = price_interpolation + jump_model.vola*S_t_0*brownian_part

    return S_t

def compute_survival_probability(S_t_0, S_t_1, delta_t,
                                 S_t_b, jump_model, barrier_option):
    """
    This function computes the probability that S crosses the barrier in the
    interval [t_0, t_1].
    Args:
        S_t_0: This float specifies the stock price at t_0.
        S_t_1: This float specifies the stock price at t_1.
        delta_t: This float specifies the length of interval [t_0, t_1].
        S_t_b: This float specifies the stock price which is used to compute
        the diffusion term bn.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        barrier_option: This class contains the parameters for the used
        barrier option.

    Returns:
        A float that specifies the survival probability.
    """
    up = 1 if barrier_option.up else 0
    diff_1 = max((S_t_0 - barrier_option.barrier) * (-1)**up, 0)
    diff_2 = max((S_t_1 - barrier_option.barrier) * (-1)**up, 0)

    b_n = jump_model.vola * S_t_b

    survival_prob = 1 - np.exp(-2 * diff_1 * diff_2 /
                               (b_n**2 * delta_t))

    return survival_prob

def milstein_fixed_grid_barrier(T, M, S_0, jump_model,
                                barrier_option):
    """
    This function implements the Milstein discretization scheme for a fixed
    time grid.
    Args:
        T: This float specifies the length of the interval.
        M: This integer specifies the number of simulated steps in
        the interval.
        S_0: This float specifies the value of S in t=0.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        barrier_option: This class contains the parameters for the used
        barrier option.

    Returns:
        A 2-D array with a discretization of a jump-diffusion process, a
        coarse version of the simulated jump-diffusion process, the
        survival probabilities on the fine path, the survival probabilities
        for the coarse path between [t_0, t'], and the survival
        probabilities on the coarse path between [t', t_1).
    """
    delta_t = T / M
    mu_fine = jump_model.drift_fine
    mu_coarse = jump_model.drift_coarse
    std = jump_model.vola

    jump_diffusion_process = [[S_0, S_0, 1, 1]]

    delta_B_coarse = 0
    jump_times_sum = 0
    jumps_occurred = []
    jumps_coarse = []

    for t in range(M):
        delta_B = np.sqrt(delta_t) * np.random.normal()
        delta_B_coarse += delta_B

        # Simulate change of S caused by diffusion process
        S_t_0_fine = jump_diffusion_process[-1][0]
        S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * delta_t + std * S_t_0_fine * \
                     delta_B

        # Simulate change of S caused by Ito correction
        correction_fine = 0.5 * std ** 2 * S_t_0_fine * \
                          (delta_B**2 - delta_t)
        S_t_1_fine += correction_fine

        survival_prob_fine = compute_survival_probability(
            S_t_0_fine, S_t_1_fine, delta_t, S_t_0_fine,
            jump_model, barrier_option)

        # Simulate change of S caused by jump process
        while jump_times_sum <= delta_t:
            jump_times_sum += jump_model.simulate_jump_time()
            jump_size, _, _ = jump_model.simulate_jump_size()
            jumps_occurred.append(jump_size)
            jumps_coarse.append(jump_size)

        jump_times_sum = jump_times_sum - delta_t
        jumps_occurred_arr = np.array(jumps_occurred[:-1])

        if len(jumps_occurred_arr) > 0:
            jumps_occurred = [jumps_occurred[-1]]

        jump_increments = np.sum(jumps_occurred_arr * S_t_0_fine)
        S_t_1_fine += jump_increments

        # Check if point is contained in coarse path and only
        # simulate change in S (for coarse path) in this case
        if (t + 1) % 2 == 0:
            # # Simulate change of S caused by diffusion process
            S_t_0_coarse = jump_diffusion_process[-2][1]
            S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * 2 * delta_t + std * S_t_0_coarse * \
                           delta_B_coarse

            # Simulate change of S caused by Ito correction
            correction_coarse = 0.5 * std ** 2 * S_t_0_coarse * \
                                (delta_B_coarse ** 2 - 2 * delta_t)
            S_t_1_coarse += correction_coarse

            # Evaluate interpolant
            S_t_intermediate = evaluate_brownian_interpolant(
                S_t_0_coarse, S_t_1_coarse, (t - 1)*delta_t, (t + 1)*delta_t,
                delta_B_coarse - delta_B, delta_B_coarse, t*delta_t,
                jump_model)

            # Calculate surviving probabilities for coarse path
            survival_prob_coarse_current = compute_survival_probability(
                S_t_0_coarse, S_t_intermediate, delta_t, S_t_0_coarse,
                jump_model, barrier_option)
            survival_prob_coarse_previous = compute_survival_probability(
                S_t_intermediate, S_t_1_coarse, delta_t, S_t_0_coarse,
                jump_model, barrier_option)
            survival_prob_coarse = survival_prob_coarse_current * \
                                   survival_prob_coarse_previous

            # Simulate change of S caused by jump process
            jumps_occurred_arr_coarse = np.array(jumps_coarse[:-1])
            jumps_coarse = [jumps_coarse[-1]]
            jump_increments_coarse = np.sum(jumps_occurred_arr_coarse * S_t_0_coarse)
            S_t_1_coarse += jump_increments_coarse

            delta_B_coarse = 0

        else:
            S_t_1_coarse = np.nan
            survival_prob_coarse = 1

        jump_diffusion_process.append([S_t_1_fine, S_t_1_coarse,
                                       survival_prob_fine,
                                       survival_prob_coarse])

    return np.array(jump_diffusion_process)

def milstein_jump_adapted_barrier(T, M, S_0, jump_model,
                                  barrier_option, cutoff_l_0, cutoff_l_1):
    """
    This function implements the jump adapted Milstein discretization
    scheme.
    Args:
        T: This float specifies the length of the interval.
        M: This integer specifies the number of simulated steps in
        the interval.
        S_0: This float specifies the value of S in t=0.
        jump_model: This class specifies which jump model is used, and
        it contains its corresponding parameters (drift, vola etc.)
        barrier_option: This class contains the parameters for the used
        barrier option.
        cutoff_l_0: This float specifies how big jumps
        (in absolute value) must be to be explicitly simulated
        on the coarse level. If None, there is no cutoff.
        cutoff_l_1: This float specifies how big jumps
        (in absolute value) must be to be explicitly simulated
        on the fine level. If None, there is no cutoff.

    Returns:
        A 2-D array with a discretization of a jump-diffusion process, a
        coarse version of the simulated jump-diffusion process, the
        survival probabilities on the fine path, the survival probabilities
        for the coarse path between [t_0, t'], and the survival
        probabilities on the coarse path between [t', t_1).
    """
    mu_fine = jump_model.drift_fine
    mu_coarse = jump_model.drift_coarse
    std = jump_model.vola

    # Initialize algorithm
    fixed_grid = np.linspace(0, T, M+1)
    jump_diffusion_process = [[S_0, S_0, 1, 1]]

    # Initialize first jump
    delta_jump_fine = jump_model.simulate_jump_time()
    jump_time_sum = delta_jump_fine
    delta_jump_coarse = delta_jump_fine
    last_simulated_jump_time_fine = 0
    last_simulated_jump_time_coarse = 0

    # Initialize path variables
    S_t_0_fine = jump_diffusion_process[-1][0]
    S_t_0_coarse = jump_diffusion_process[-1][1]

    intermediate_brownian_increments = []
    intermediate_time_stamps = [0]

    for j in range(1, M+1):
        t_1 = fixed_grid[j]
        t_0 = fixed_grid[j-1]

        # Simulate jumps and S between t_0 and t_1
        while jump_time_sum < t_1:
            jump_size, simulated_fine, simulated_coarse = (
                jump_model.simulate_jump_size(cutoff_l_0, cutoff_l_1))

            if simulated_fine:
                # Simulate change in fine path
                # Drift and diffusion
                delta_B = np.sqrt(delta_jump_fine) * np.random.normal()
                intermediate_brownian_increments.append(delta_B)
                intermediate_time_stamps.append(jump_time_sum)

                S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * delta_jump_fine + std * S_t_0_fine * delta_B

                # Ito correction
                S_t_1_fine += 0.5 * std ** 2 * S_t_0_fine * (delta_B ** 2 - delta_jump_fine)

                survival_prob_fine = compute_survival_probability(
                    S_t_0_fine, S_t_1_fine, delta_jump_fine, S_t_0_fine,
                    jump_model, barrier_option)

                # Jump
                S_t_1_fine += S_t_1_fine * jump_size

                # Update variables on fine path
                S_t_0_fine = S_t_1_fine
                last_simulated_jump_time_fine = jump_time_sum
                delta_jump = jump_model.simulate_jump_time()
                delta_jump_fine = delta_jump

                if simulated_coarse:
                    # Simulate change in coarse path
                    # Drift and diffusion
                    delta_B_coarse = np.sum(intermediate_brownian_increments)
                    S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * delta_jump_coarse + \
                                   std * S_t_0_coarse * delta_B_coarse

                    # Ito correction
                    S_t_1_coarse += 0.5 * std ** 2 * S_t_0_coarse * (delta_B_coarse ** 2 - delta_jump_coarse)

                    # Calculate surviving probabilities for coarse path
                    # If there are previous time stamps that are not in the coarse path
                    # use brownian interpolant to compute survival probabilities
                    if len(intermediate_brownian_increments) > 1:
                        survival_prob_coarse = 1

                        t_0_intermediate = intermediate_time_stamps[0]
                        t_1_intermediate = intermediate_time_stamps[-1]
                        S_t_intermediate_previous = S_t_0_coarse

                        for k in range(len(intermediate_brownian_increments[:-1])):
                            increment = np.sum(intermediate_brownian_increments[:(k+1)])
                            t = intermediate_time_stamps[k + 1]

                            S_t_intermediate_current = evaluate_brownian_interpolant(
                                S_t_0_coarse, S_t_1_coarse, t_0_intermediate, t_1_intermediate,
                                increment, delta_B_coarse, t, jump_model)
                            dt_intermediate = t - intermediate_time_stamps[k]

                            survival_prob_coarse_current = compute_survival_probability(
                                S_t_intermediate_previous, S_t_intermediate_current, dt_intermediate,
                                S_t_0_coarse, jump_model, barrier_option)
                            survival_prob_coarse *= survival_prob_coarse_current

                            S_t_intermediate_previous = S_t_intermediate_current

                        survival_prob_coarse_current = compute_survival_probability(
                            S_t_intermediate_previous, S_t_1_coarse,
                            t_1_intermediate - intermediate_time_stamps[-2],
                            S_t_0_coarse, jump_model, barrier_option)
                        survival_prob_coarse *= survival_prob_coarse_current

                    else:
                        survival_prob_coarse = compute_survival_probability(
                            S_t_0_coarse, S_t_1_coarse, delta_jump_coarse, S_t_0_coarse,
                            jump_model, barrier_option)

                    # Jump
                    S_t_1_coarse += S_t_1_coarse * jump_size

                    # Update variables for coarse path
                    S_t_0_coarse = S_t_1_coarse
                    delta_jump_coarse = delta_jump
                    last_simulated_jump_time_coarse = jump_time_sum
                    intermediate_brownian_increments = []
                    intermediate_time_stamps = [intermediate_time_stamps[-1]]

                else:
                    S_t_1_coarse = np.nan
                    delta_jump_coarse += delta_jump
                    survival_prob_coarse = 1

                # Append results and update variables
                jump_diffusion_process.append([S_t_1_fine, S_t_1_coarse,
                                               survival_prob_fine,
                                               survival_prob_coarse])

            else:
                delta_jump = jump_model.simulate_jump_time()
                delta_jump_fine += delta_jump
                delta_jump_coarse += delta_jump

            jump_time_sum = delta_jump + jump_time_sum

        # Simulate S between last jump and t_1
        # Drift and diffusion
        dt_fine = t_1 - max(t_0, last_simulated_jump_time_fine)
        delta_B = np.sqrt(dt_fine) * np.random.normal()
        intermediate_brownian_increments.append(delta_B)
        intermediate_time_stamps.append(t_1)

        S_t_1_fine = S_t_0_fine + mu_fine * S_t_0_fine * dt_fine + std * S_t_0_fine * delta_B

        # Ito correction
        S_t_1_fine += 0.5 * std ** 2 * S_t_0_fine * (delta_B ** 2 - dt_fine)

        survival_prob_fine = compute_survival_probability(
            S_t_0_fine, S_t_1_fine, dt_fine, S_t_0_fine, jump_model, barrier_option)

        # Check if point is contained in coarse path and only
        # simulate change in S (for coarse path) in this case
        if j % 2 == 0:
            dt_coarse = t_1 - max(fixed_grid[j-2], last_simulated_jump_time_coarse)
            # Simulate S in t_1
            # Drift and diffusion
            delta_B_coarse = np.sum(intermediate_brownian_increments)
            S_t_1_coarse = S_t_0_coarse + mu_coarse * S_t_0_coarse * dt_coarse + \
                           std * S_t_0_coarse * delta_B_coarse

            # Ito correction
            S_t_1_coarse += 0.5 * std ** 2 * S_t_0_coarse * (delta_B_coarse ** 2 - dt_coarse)

            # Calculate surviving probabilities for coarse path
            # If there are previous time stamps that are not in the coarse path
            # use brownian interpolant to compute survival probabilities
            if len(intermediate_brownian_increments) > 1:
                survival_prob_coarse = 1

                t_0_intermediate = intermediate_time_stamps[0]
                t_1_intermediate = intermediate_time_stamps[-1]
                S_t_intermediate_previous = S_t_0_coarse

                for k in range(len(intermediate_brownian_increments[:-1])):
                    increment = np.sum(intermediate_brownian_increments[:(k + 1)])
                    t = intermediate_time_stamps[k + 1]

                    S_t_intermediate_current = evaluate_brownian_interpolant(
                        S_t_0_coarse, S_t_1_coarse, t_0_intermediate, t_1_intermediate,
                        increment, delta_B_coarse, t, jump_model)

                    dt_intermediate = t - intermediate_time_stamps[k]
                    survival_prob_coarse_current = compute_survival_probability(
                        S_t_intermediate_previous, S_t_intermediate_current, dt_intermediate,
                        S_t_0_coarse, jump_model, barrier_option)

                    survival_prob_coarse *= survival_prob_coarse_current

                    S_t_intermediate_previous = S_t_intermediate_current

                survival_prob_coarse_current = compute_survival_probability(
                    S_t_intermediate_previous, S_t_1_coarse,
                    t_1_intermediate - intermediate_time_stamps[-2],
                    S_t_0_coarse, jump_model, barrier_option)
                survival_prob_coarse *= survival_prob_coarse_current

            else:
                survival_prob_coarse = compute_survival_probability(
                    S_t_0_coarse, S_t_1_coarse, dt_coarse, S_t_0_coarse,
                    jump_model, barrier_option)

            # Update variables
            S_t_0_coarse = S_t_1_coarse
            delta_jump_coarse = jump_time_sum - t_1
            intermediate_brownian_increments = []
            intermediate_time_stamps = [intermediate_time_stamps[-1]]

        else:
            S_t_1_coarse = np.nan
            survival_prob_coarse = 1

        # Append results and update variables
        jump_diffusion_process.append([S_t_1_fine, S_t_1_coarse,
                                       survival_prob_fine,
                                       survival_prob_coarse])

        S_t_0_fine = S_t_1_fine
        delta_jump_fine = jump_time_sum - t_1

    return np.array(jump_diffusion_process)
