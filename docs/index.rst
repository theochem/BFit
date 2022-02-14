.. BFit documentation master file, created by
   sphinx-quickstart on Mon Nov 22 16:46:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BFit's documentation!
================================

`BFit <https://github.com/theochem/bfit>`_ is a free, open-source, and cross-platform
Python library for fitting a convex sum of Gaussian functions of s-type and p-type to any
probability distribution.

Installation can be found in the `github/README <https://github.com/theochem/bfit>`_ and
contains a example of using the Kullback-Leibler fixed point iteration method.

For more examples of using BFit, see the interactive
`Jupyter binder <https://mybinder.org/v2/gh/theochem/bfit/master?labpath=%2Fexamples%2>`_
or files in the `example folder <https://github.com/theochem/BFit/tree/master/examples>`_
to see specific examples on how to fit using the different algorithms and objective
functions.  The table below gives a brief description of each module for which you can look
through the API and the paper provides specific details.

Please use the following citation in any publication using BFit library:

    **"BFit: Information-Theoretic Approach to Basis-Set Fitting of Electron Densities."**,
    A. Tehrani, F. Heidar-Zadeh, J. S. M. Anderson, T. Verstraelen, R. Cuevas-Saavedra,
    I. Vinogradov, D. Chakraborty, P. W. Ayers


BFIt is released under the
`GNU General Public License v3.0 <https://github.com/theochem/bfit/blob/master/LICENSE>`_.
Please report any issues you encounter while using BFit library on
`GitHub Issues <https://github.com/theochem/bfit/issues>`_.
For further information and inquiries please contact us at `qcdevs@gmail.com <qcdevs@gmail.com>`_.




.. list-table:: **Module Description**
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - *measure.py*
     - Specifies the objective functions: Kullback-Leibler Divergence, Squared Difference, Tsallis-Divergence.
   * - *model.py*
     - Specifies the model for fitting: Univariate Gaussian distribution composed of s-type and p-type Gaussians and
       three-dimensional Gaussian distribution centered at different locations.
   * - *grid.py*
     - Contains the grid for integration and defining the points.
   * - *fit.py*
     - Contains the fitting algorithms: Kullback-Leibler fixed point method and ScipyFit that
       uses SLSQP and trust-constraint method found in `scipy.optimize`.
   * - *greedy.py*
     - Contains the greedy algorithm of iteratively selectiveling the next set of basis-functions
       from the previous set.
   * - *density.py*
     - Provides atomic, and kinetic densities of atoms from Hatree-Fock wavefunctions composed of
       Slater-type orbitals. See the data folder for more details on the wavefunctions.
   * - *parse_ugbs.py*
     - Obtain the universal Gaussian basis-set exponents for each atom.

.. toctree::
   :maxdepth: 4
   :caption: API Documentation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
