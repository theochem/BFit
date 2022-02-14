r"""
This files generates the results for Table (3) in the BFit paper.

Specifically, it optimizes the spherically-averaged Kullback-Leibler divergence
 function using the SLSQP function (from class ScipyFit) of the atomic density.
 The initial guess is chosen to be universal Gaussian basis-set (UGBS) multipled
 by two. Normalized Gaussian model is used.
"""
import numpy as np

from bfit.density import SlaterAtoms
from bfit.fit import ScipyFit, KLDivergenceFPI
from bfit.grid import ClenshawRadialGrid
from bfit.measure import KLDivergence
from bfit.model import AtomicGaussianDensity
from bfit.parse_ugbs import get_ugbs_exponents


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
][-1:]:
    print("\n Start Atom %s" % element)

    # Construct Integration Grid
    grid = ClenshawRadialGrid(
        atomic_numb, num_core_pts=10000, num_diffuse_pts=899, extra_pts=[50, 75, 100]
    )

    # Construct Initial Guess with UGBS
    ugbs = get_ugbs_exponents(element)
    exps_s = ugbs["S"]
    num_s = len(exps_s)
    exps_p = ugbs["P"]
    num_p = len(exps_p)
    c0 = np.array([atomic_numb / (num_s + num_p)] * (num_s + num_p))
    e0 = np.array(exps_s + exps_p) * 2.0

    model = AtomicGaussianDensity(grid.points, num_s=num_s, num_p=num_p, normalize=True)
    atomic_dens = SlaterAtoms(element=element)
    density = atomic_dens.atomic_density(grid.points)

    # Construct Fitting Object using SLSQP and optimizing KL
    measure = KLDivergence(mask_value=1e-18)
    fit_KL_slsqp = ScipyFit(grid, density, model, measure=measure, method="SLSQP", spherical=True)

    # Run the SLSQP optimization algorithm
    results = fit_KL_slsqp.run(c0, e0, maxiter=10000, disp=True, with_constraint=True, tol=1e-14)

    print("SLSQP INFO")
    print("----------")
    print("Success %s" % results["success"])
    print("Final Coeffs")
    print(results["coeffs"])
    print("Final Exponents")
    print(results["exps"])
    print("Integration Value & L1 & L_infinity & LS & KL (With 4 pi r^2 included)")
    p = results["performance"]
    print(p)
