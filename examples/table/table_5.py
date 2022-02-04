r"""
This files generates the results for Table (5) in the BFit paper.

Specifically, it optimizes the spherically-averaged Kullback-Leibler divergence
 function using the SLSQP function (from class ScipyFit) of the
 positive definite kinetic energy density.
 The initial guess is chosen to be universal Gaussian basis-set (UGBS) multipled
 by (10/3). Normalized Gaussian model is used.
"""
import numpy as np

from bfit.density import SlaterAtoms
from bfit.fit import KLDivergenceFPI
from bfit.grid import ClenshawRadialGrid
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
][6:]:
    print("\nElement %s " % element)
    # Define Grid
    grid = ClenshawRadialGrid(
        atomic_numb, num_core_pts=10000, num_diffuse_pts=899, extra_pts=[50, 75, 100],
        include_origin=False
    )

    # Number of S-orbitals is zero, Number of P-orbitals is
    ugbs = get_ugbs_exponents(element)
    exps_s = ugbs["S"]
    num_s = len(exps_s)
    exps_p = ugbs["P"]
    num_p = len(exps_p)

    # Define Model
    model = AtomicGaussianDensity(grid.points, num_s=num_s, num_p=num_p, normalize=True)

    # Atomic Density
    dens_obj = SlaterAtoms(element)
    kinetic_energy = dens_obj.positive_definite_kinetic_energy(grid.points)
    print(f"Integrated kinetic energy {kinetic_energy} and "
          f"Actual Kinetic Energy {dens_obj.kinetic_energy}")

    # Fit object
    fit = KLDivergenceFPI(grid, kinetic_energy, model, mask_value=1e-18, spherical=True)

    # Initial Guess
    c0 = np.array([atomic_numb / (num_s + num_p)] * (num_s + num_p))
    e0 = np.array(exps_s + exps_p)
    e0 *= (10.0 / 3.0)

    # Run It
    results = fit.run(c0, e0, maxiter=10000, c_threshold=1e-6, e_threshold=1e-6, d_threshold=1e-14,
                      disp=True)

    # Results
    print("MBIS INFO\n---------")
    print("Success %s" % results["success"])
    print("Final Coeffs")
    print(results["coeffs"])
    print("Final Exponents")
    print(results["exps"])
    print("Integration Value & L1 & L_infinity & LS & KL (With 4 pi r^2 included)")
    final_result = results["performance"][-1]
    print(final_result)
