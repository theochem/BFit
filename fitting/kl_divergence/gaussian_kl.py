r"""
This Contains the mbis class responsible for fitting to a gaussian basis set.

Contains the Minimal-Basis-Set-Inter - Stockholder algorithm.
This algorithm minizes the Kullback-Leibler between a probability distribution
composed of gaussian basis set and any other probability distribution.

Note that being a probability distribution means it is integrable.
"""


from __future__ import division
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting
import numpy as np
import numpy.ma as ma

__all__ = ["GaussianKullbackLeibler"]


class GaussianKullbackLeibler(KullbackLeiblerFitting):
    r"""

    """
    def __init__(self, grid_obj, true_model, inte_val=None):
        r"""

        Parameters
        ----------


        """
        super(GaussianKullbackLeibler, self).__init__(grid_obj, true_model, inte_val)
        self.grid_points = ma.asarray(np.reshape(grid_obj.radii, (len(grid_obj.radii), 1)))
        # TODO: Seems like I don't need this attribute
        self.masked_grid_squared = ma.asarray(np.power(self.grid_obj.radii, 2.))

    def get_norm_coeffs(self, coeff_arr, exp_arr):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        return coeff_arr * self.get_norm_consts(exp_arr)

    def get_model(self, coeff_arr, exp_arr, norm=True):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        exponential = np.exp(-exp_arr * np.power(self.grid_points, 2.))
        if norm:
            coeff_arr = self.get_norm_coeffs(coeff_arr, exp_arr)
        normalized_gaussian_density = np.dot(exponential, coeff_arr)
        # TODO: See if this really is necessarry
        #index_where_zero_occurs = np.where(normalized_gaussian_density == 0.)
        #if len(index_where_zero_occurs[0]) != 0:
        #    normalized_gaussian_density[index_where_zero_occurs] = \
        #            normalized_gaussian_density[index_where_zero_occurs[0][0] - 1]
        return normalized_gaussian_density

    def get_inte_factor(self, exponent, masked_normed_gaussian, upt_exponent=False):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        ratio = self.ma_true_mod / masked_normed_gaussian
        grid_squared = self.grid_obj.radii**2.
        integrand = ratio * np.ma.asarray(np.exp(-exponent * grid_squared))
        if upt_exponent:
            integrand = integrand * self.masked_grid_squared
            return self._get_norm_constant(exponent) * self.grid_obj.integrate_spher(integrand)
        return self._get_norm_constant(exponent) * self.grid_obj.integrate_spher(integrand)

    def _update_coeffs_gauss(self, coeff_arr, exp_arr):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        gaussian = ma.asarray(self.get_model(coeff_arr, exp_arr))
        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_inte_factor(exp_arr[i], gaussian)
        return new_coeff / self.lagrange_multiplier

    def _update_func_params(self, coeff_arr, exp_arr, with_convergence=True):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        masked_normed_gaussian = np.ma.asarray(self.get_model(coeff_arr, exp_arr)).copy()

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            if with_convergence:
                new_exps[i] = 3. * self._lagrange_multiplier
            else:
                new_exps[i] = 3. * self.get_inte_factor(exp_arr[i], masked_normed_gaussian)
            integration = self.get_inte_factor(exp_arr[i], masked_normed_gaussian, upt_exponent=True)
            new_exps[i] /= (2. * integration)
        return new_exps

    def _get_norm_constant(self, exponent):
        return (exponent / np.pi) ** (3./2.)

    def _update_coeffs(self, coeff_arr, exp_arr):
        new_coeff = self._update_coeffs_gauss(coeff_arr, exp_arr)
        return new_coeff, coeff_arr

    def _update_exps(self, coeff_arr, exp_arr):
        new_exps = self._update_func_params(coeff_arr, exp_arr)
        return new_exps, exp_arr

    def _get_deriv_coeffs(self, coeffs, fparams):
        pass

    def _get_deriv_fparams(self, coeffs, fparams):
        pass
