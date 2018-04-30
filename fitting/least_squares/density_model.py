r"""
This file designates how to define your model (e.g. gaussian basis set) to
fit to slater densities using the least squares optimization.

Hence the DensityModel is the abstract class for all models.
It also contains standard error measures to be used when fitting to get a sense
of how good the fit is.
"""

import abc
import numpy as np
from fitting.gbasis.gbasis import UGBSBasis

__all__ = ["DensityModel"]
#TODO: Remove GBASIS and UGBS, because even tempered seems better.


class DensityModel(object):
    """
    This is an abstract class for the gaussian density model
    used for fitting slater densities.

    Primarily used to define the cost function/objective function and
    the residual which is being minimized through least squares.
    Additionally, contains tools to define different error measures,
    as well as UGBS exponents used to define proper initial guesses
    for the Gaussian density model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid, element=None, electron_density=None):
        r"""

        Parameters
        ----------
        grid : np.ndarray
        element : str
        electron_density : np.ndarray

        Raises
        ------
        TypeError
            If an argument of an invalid type is used

        """
        if element is not None and not isinstance(element, str):
            raise TypeError("Element name should be string.")
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if electron_density is not None:
            if not isinstance(electron_density, np.ndarray):
                raise TypeError("Electron density should be an array.")
            if grid.shape != electron_density.shape:
                raise ValueError("Electron density and _grid should be the same "
                                 "size.")
        self._element = element
        self._grid = np.ravel(np.copy(grid))
        self._electron_density = np.ravel(electron_density)

        if element is not None:
            self._element = element.lower()
            gbasis = UGBSBasis(element)
            self._UGBS_s_exponents = 2.0 * gbasis.exponents('s')
            self._UGBS_p_exponents = 2.0 * gbasis.exponents('p')
            if self._UGBS_p_exponents.size == 0.0:
                self._UGBS_p_exponents = np.copy(self._UGBS_s_exponents)

    @property
    def element(self):
        return self._element

    @property
    def grid(self):
        return self._grid

    @property
    def electron_density(self):
        return self._electron_density

    @property
    def UGBS_s_exponents(self):
        return self._UGBS_s_exponents

    @property
    def UGBS_p_exponents(self):
        return self._UGBS_p_exponents

    @abc.abstractmethod
    def create_model(self):
        """
        """
        raise NotImplementedError("Need to implement the density model")

    @abc.abstractmethod
    def cost_function(self):
        """
        """
        raise NotImplementedError("Need to implement the cost function")

    @abc.abstractmethod
    def derivative_of_cost_function(self):
        """
        """
        raise NotImplementedError("Need to Implement the derivative of cost "
                                  "function")

    @abc.abstractmethod
    def create_cofactor_matrix(self):
        pass

    def calculate_residual(self, *args):
        res = self._electron_density - self.create_model(*args)
        return res

    """
    def calculate_residual_based_on_core(self, *args):
        residual = np.ravel(self.electron_density_core) - self.density_model.create_model(*args)
        return residual

    def calculate_residual_based_on_valence(self, *args):
        residual = np.ravel(self.electron_density_valence) - self.density_model.create_model(*args)
        return residual
    """

    def integrate_model_trapz(self, approx_model):
        r"""
        Integrates the approximate model with an added r^2 over [0, inf),
        using the trapezoidal method.

        Parameters
        ----------
        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        int : float
              Integration value of r^2 with approximate model over the _grid.
        """
        grid_squared = np.ravel(self._grid ** 2.)
        int = np.trapz(y=grid_squared * approx_model, x=self._grid)
        return int

    def get_error_diffuse(self, true_model, approx_model):
        r"""
        This error measures how good the mbis is between the approximate and
        true density at long densities.
        # TODO: FIX THIS TO MAKE THE LATEX INLINE
        ..math::
            Given two functions, denoted f, g.
            The error is given as \int r^2 |f(r) - g(r)|dr

        Parameters
        ----------
        true_model : np.ndarray
                     The true model that is being fitted.

        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        error : float
                A positive real number that measures how good the mbis is.
        """
        abs_diff = np.absolute(np.ravel(true_model) - np.ravel(approx_model))
        error = np.trapz(y=self._grid**2 * abs_diff, x=self._grid)
        return error

    def get_integration_error(self, true_model, approx_model):
        r"""
        This error measures the difference in integration of the two models.

        ..math::
            Given two functions, denoted f, g.
            The error is given as |\int r^2f(r)dr - \int r^2g(r)dr|.

        Parameters
        ----------
        true_model : np.ndarray
                     The true model that is being fitted.

        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        error : float
                Measures the difference in integration of the two models.
        """
        integrate_true_model = self.integrate_model_trapz(true_model)
        integrate_approx_model = self.integrate_model_trapz(approx_model)
        diff_model = integrate_true_model - integrate_approx_model
        return np.absolute(diff_model)

    def generation_of_UGBS_exponents(self, p, UGBS_exponents):
        r"""
        #TODO: Implement Proper Documentation for This.
        Generates Universal Gaussian Basis Sets (UGBS) exponents.
        Used for Greedy I THINK
        Parameters
        ----------
        p : int

        UGBS_exponents : array

        Returns
        -------
        array

        """
        max_ugbs= np.amax(UGBS_exponents)
        min_ugbs = np.amin(UGBS_exponents)

        def get_numb_gauss_funcs(p, max, min):
            num_of_basis_functions = np.log(2 * max / min) / np.log(p)
            return num_of_basis_functions

        numb_basis_funcs = int(get_numb_gauss_funcs(p, max_ugbs, min_ugbs))
        new_gauss_exps = np.array([min_ugbs])
        for n in range(1, numb_basis_funcs + 1):
            next_exponent = min_ugbs * np.power(p, n)
            new_gauss_exps = np.append(new_gauss_exps, next_exponent)

        return new_gauss_exps
