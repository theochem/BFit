r"""
This files generates the results for Table (4) in the BFit paper.

Specifically, it optimizes the least-squares (spherically-averaged) of the
 atomic density.  The un-normalized Gaussian model is used and the choice
 of adding the constraint that the integral of the model density should
 be equal to the integral of the atomic density is used with the variable
 `WITH_CONSTRAINT`. The universal Gaussian basis set exponents multipled by
 two are used as the initial guess.

"""
import numpy as np

from bfit.density import SlaterAtoms
from bfit.grid import ClenshawRadialGrid
from bfit.model import AtomicGaussianDensity
from bfit.measure import SquaredDifference
from bfit.fit import ScipyFit
from bfit.parse_ugbs import get_ugbs_exponents

WITH_CONSTRAINT = False

for atomic_numb, element in [
    (2, "he"),
    (4, "be"),
    (9, "f"),
    (13, "al"),
    (19, "k"),
    (22, "ti"),
    (30, "zn"),
    (45, "rh"),
    (53, "i")
][8:]:
    print("Element %s " % element)

    # Define Integration Grid
    grid = ClenshawRadialGrid(
        atomic_numb, num_core_pts=10000, num_diffuse_pts=899, extra_pts=[50, 75, 100]
    )

    # Use the same number of s-type and p-type functions as UGBS
    ugbs = get_ugbs_exponents(element)
    exps_s = ugbs["S"]
    exps_p = ugbs["P"]
    num_s = len(exps_s)
    num_p = len(exps_p)

    # Define Un-normalized Gaussian Model
    model = AtomicGaussianDensity(grid.points, num_s=num_s, num_p=num_p, normalize=False)

    # Atomic Density
    dens_obj = SlaterAtoms(element)
    density = dens_obj.atomic_density(grid.points)

    # Fit object with least-squares as the objective function.
    measure = SquaredDifference()
    fit = ScipyFit(grid, density, model, measure=measure, method="SLSQP", spherical=True)

    # Define UGBS initial guess
    c0 = np.array([atomic_numb / (num_s + num_p)] * (num_s + num_p))
    e_0 = exps_s + exps_p
    e_0 = np.array(e_0)
    e_0 *= 2.0

    # Run the optimization algorithm
    results = fit.run(
        c0, e_0, maxiter=10000, disp=True, with_constraint=WITH_CONSTRAINT, tol=1e-14
    )

    print("SLSQP INFO")
    print("---------")
    print("Success %s" % results["success"])
    print("Final Coeffs")
    print(results["coeffs"])
    print("Final Exponents")
    print(results["exps"])
    print("Integration Value & L1 & L_infinity & LS & KL (With 4 pi r^2 included)")
    p = results["performance"]
    print(p)
