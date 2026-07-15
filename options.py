"""
Tools to model different options.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import numpy as np
import pandas as pd


class Option:
    """
    Class that is used to save general information about options.

    Initialized with:
    - an initial price
    - a strike price
    - a maturity date
    - a risk-free rate
    - a jump model
    """

    def __init__(self, S_0, strike, maturity, r, jump_model):
        """
        This is the init method for the class 'Option'.
        Args:
            S_0: This float specifies the value of S in t=0.
            strike: This float represent the strike of the option.
            history of the underlying including the price, corresponding
            timestamps, the diffusion process and an info if a jump occurred
            at the given timestamp.
            maturity: This float specifies the maturity date of the
            option.
            r: This float specifies the assumed risk-free rate
            jump_model: This class specifies which jump model is used.
        """
        self.S_0 = S_0
        self.strike = strike
        self.maturity = maturity
        self.r = r
        self.jump_model = jump_model

    def payoff_fine(self, simulated_path):
        """
        This method calculates the option's payoff at maturity on the fine
        level. It is implemented by the subclasses of 'Option'.
        Args:
            simulated_path: This 2-D array contains a simulated fine and coarse path
            together with additional required information to correctly compute the
            payoff of an option.
        """
        pass

    def payoff_coarse(self, simulated_path):
        """
        This method calculates the option's payoff at maturity on the coarse
        level. It is implemented by the subclasses of 'Option'.
        Args:
            simulated_path: This 2-D array contains a simulated fine and coarse path
            together with additional required information to correctly compute the
            payoff of an option.
        """
        pass


class KnockOutOption(Option):
    """
    This class implements a knock-out option. It can be used to simulate
    up-and-out and down-and-out options.

    Initialized with:
    - a barrier
    - a classifier that specifies if barrier is upper or lower bound
    """

    def __init__(self, S_0, strike, maturity, r, jump_model, barrier, up):
        """
        This is the init method for the class 'KnockOutOption'.
        Args:
            S_0:  Defined in parent class.
            strike: Defined in parent class.
            maturity: Defined in parent class.
            r: Defined in parent class.
            jump_model: Defined in parent class.
            barrier: This flot specifies the barrier.
            up: This bool specifies if the option is an up-and-out
            or down-and-out option.
        """
        super().__init__(S_0, strike, maturity, r, jump_model)
        self.barrier = barrier
        self.up = up

    def payoff_fine(self, simulated_path):
        """
        This method calculates the payoff for the given knock-and-out option
        on the fine level.
        Args:
            simulated_path: A 2-D array with a discretization of a
            jump-diffusion process, a coarse version of the simulated
            jump-diffusion process, the survival probabilities on the fine
            path, the survival probabilities for the coarse path between
            [t_0, t'], and the survival probabilities on the coarse path
            between [t', t_1).

        Returns:
            A float that specifies the option's payoff at maturity.
        """
        p_barrier_reached = np.prod(simulated_path[:, 2])

        return np.exp(-self.r * self.maturity) * \
            max(simulated_path[-1, 0] - self.strike, 0) * \
            p_barrier_reached

    def payoff_coarse(self, simulated_path):
        """
        This method calculates the payoff for the given knock-and-out option
        on the coarse level.
        Args:
            simulated_path: A 2-D array with a discretization of a
            jump-diffusion process, a coarse version of the simulated
            jump-diffusion process, the survival probabilities on the fine
            path, the survival probabilities for the coarse path between
            [t_0, t'], and the survival probabilities on the coarse path
            between [t', t_1).

        Returns:
            A float that specifies the option's payoff at maturity.
        """
        p_barrier_reached = np.prod(simulated_path[:, 3])

        return np.exp(-self.r * self.maturity) * \
            max(simulated_path[-1, 1] - self.strike, 0) * \
            p_barrier_reached


class AsianOption(Option):
    """
    This class implements a continuous arithmetic asian option.

    Initialized with:
    - a classifier that specifies if option is a call or put
    """

    def __init__(self, S_0, strike, maturity, r, jump_model, call):
        """
        This is the init method for the class 'AsianOption'.
        Args:
            S_0:  Defined in parent class.
            strike: Defined in parent class.
            maturity: Defined in parent class.
            r: Defined in parent class.
            jump_model: Defined in parent class.
            call: This bool specifies if the option is a call or put.
        """
        super().__init__(S_0, strike, maturity, r, jump_model)
        self.call = call

    def payoff_fine(self, simulated_path):
        """
        This method calculates the payoff for the given asian option on the
        fine level.
        Args:
            simulated_path: A 2-D array with a discretization of a
            jump-diffusion process, a coarse version of the simulated
            jump-diffusion process, and the corresponding timestamps.

        Returns:
            A float that specifies the option's payoff at maturity.
        """
        # Evaluate integral over S_t by using the trapezoidal rule
        # Fixed grid
        if simulated_path.shape[1] == 3:
            h_l = simulated_path[1:, 2] - simulated_path[:-1, 2]
            sum_S = simulated_path[1:, 0] + simulated_path[:-1, 0]

        # Jump adapted
        else:
            h_l = simulated_path[1:, 4] - simulated_path[:-1, 4]
            sum_S = simulated_path[1:, 1] + simulated_path[:-1, 0]

        S_mean = 0.5 * np.dot(h_l, sum_S)
        S_mean /= self.maturity

        if self.call:
            return np.exp(-self.r * self.maturity) * \
                max(S_mean - self.strike, 0)
        else:
            return np.exp(-self.r * self.maturity) * \
                max(self.strike - S_mean, 0)

    def payoff_coarse(self, simulated_path):
        """
        This method calculates the payoff for the given asian option
        on the coarse level.
        Args:
            simulated_path: A 2-D array with a discretization of a
            jump-diffusion process, a coarse version of the simulated
            jump-diffusion process, and the corresponding timestamps.

        Returns:
            A float that specifies the option's payoff at maturity.
        """
        # Fixed grid
        if simulated_path.shape[1] == 3:
            simulated_path = simulated_path[~np.isnan(simulated_path[:, 1])]
            simulated_path[:, 0] = simulated_path[:, 1]

        # Jump adapted
        else:
            simulated_path = simulated_path[~np.isnan(simulated_path[:, 2])]

            simulated_path[:, 0] = simulated_path[:, 2]
            simulated_path[:, 1] = simulated_path[:, 3]

        return self.payoff_fine(simulated_path)
