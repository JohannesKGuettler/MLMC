"""
Tools to implement a Merton jump diffusion model.
"""

__author__ = "7552155, Johannes Guettler"
__email__ = "johannes.k.guettler@gmail.com"

import numpy as np


class MertonJD:
    """
    Class that is used to save basic information about a jump-diffusion model.

    Initialized with:
    - a drift on the fine level
    - a drift on the coarse level
    - a volatility
    - a jump rate
    - a mean jump size
    - a standard deviation of the jump size
    """

    def __init__(self, drift_fine, drift_coarse, vola, jumps_mean, jumps_std, jump_rate):
        """
        This is the init method for the class 'JumpModel'.
        Args:
            drift_coarse: This float specifies the drift of the jump diffusion
            process on the coarse level.
            drift_fine: This float specifies the drift of the jump diffusion
            process on the fine level.
            vola: This float specifies the volatility of the jump diffusion
            process.
            jump_rate: This float specifies the expected number of jumps in a
            standardized time interval.
            jumps_mean: This float specifies the mean of the logarithmic
            relative jump sizes.
            jumps_std: This float specifies the standard deviation of the
            logarithmic relative jump sizes.
        """
        self.drift_coarse = drift_coarse
        self.drift_fine = drift_fine
        self.vola = vola

        self.jump_rate = jump_rate
        self.jumps_mean = jumps_mean
        self.jumps_std = jumps_std

    def simulate_jump_time(self):
        """
        This method generates a jump time by determining the time until the
        next jump.
        Returns:
            A float that specifies how much time has to pass until the next
            jump.
        """
        return np.random.exponential(1/self.jump_rate)

    def simulate_jump_size(self, cutoff_value_l_0=None, cutoff_value_l_1=None):
        """
        This method generates a jump size for a certain jump.It is implemented
        by the subclasses of 'JumpModel'.
        Args:
            cutoff_value_l_0: This float specifies how big jumps
            (in absolute value) must be to be explicitly simulated
            on the coarse level. If None, there is no cutoff.
            cutoff_value_l_1: This float specifies how big jumps
            (in absolute value) must be to be explicitly simulated
            on the fine level. If None, there is no cutoff.

        Returns:
            A float that specifies the jump size.
            A bool that specifies if the jump was above or below the cutoff
            value for the fine level.
            A bool that specifies if the jump was above or below the cutoff
            value for the coarse level.
        """
        above_fine = True
        above_coarse = True

        Y = np.random.normal(self.jumps_mean, self.jumps_std)

        # Check if jump exceeded threshold
        if cutoff_value_l_1 is not None and cutoff_value_l_1 >= abs(Y):
            above_fine = False
        if cutoff_value_l_0 is not None and cutoff_value_l_0 >= abs(Y):
            above_coarse = False

        return np.exp(Y) - 1, above_fine, above_coarse
