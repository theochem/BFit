r"""
This files generates the results for Table (1) and (2) in the BFit paper.

Specifically, it optimizes the Kullback-Leibler Divergence using the fixed
 point iteration method. A normalized Gaussian model is used whose initial
 guess is the universal Gaussian basis-set multipled by two.  The constraint
 that the integral of the model should equal the atomic number is added, via
 the attribute `integral_dens`. The attribute `disp` displays the results
 at each iteration.
"""
import numpy as np

from bfit.density import SlaterAtoms
from bfit.fit import ScipyFit,KLDivergenceFPI
from bfit.grid import ClenshawRadialGrid
from bfit.model import AtomicGaussianDensity
from bfit.measure import KLDivergence
from bfit.parse_ugbs import get_ugbs_exponents
from atomdb import load

results_final = {}
atoms = ["H",  "C", "N", "O", "F", "P", "S", "Cl"]
atomic_numbs = [1, 6, 7, 8, 9, 15, 16, 17] #[1 + i for i in range(0, len(atoms))]
mult = [2, 3, 4, 3, 2, 4, 3, 2]
for k, element in enumerate(atoms):
    print("Start Atom %s" % element)

    # Construct a integration grid
    atomic_numb = atomic_numbs[k]
    grid = ClenshawRadialGrid(
        atomic_numb, num_core_pts=10000, num_diffuse_pts=899, include_origin=True#, extra_pts=[50,75,100],
    )

    # Initial Guess constructed from UGBS
    ugbs = get_ugbs_exponents(element)
    exps_s = ugbs["S"]
    num_s = len(exps_s)
    exps_p = ugbs["P"]
    num_p = len(exps_p)
    coeffs = np.array([atomic_numb / (num_s + num_p)] * (num_s + num_p))
    e_0 = np.array(exps_s + exps_p) * 2.0

    # Construct Atomic Density and Fitting Object
    #density = SlaterAtoms(element=element).atomic_density(grid.points)
    # Use AtomDB to calculate the density
    atom = load(elem=element, charge=0, mult=mult[k], dataset="hci", datapath="/home/ali-tehrani/SoftwareProjects/AtomDBdata")

    dens = atom.dens_func()
    density = dens(grid.points)

    import matplotlib.pyplot as plt
    # plt.plot(grid.points, density, "bo-")
    # plt.show()
    print(density[-1], grid.points[-1], grid.points[0:3])
    density[density < 0.0] = 0.0
    # continue
    # assert 1 == 0
    model = AtomicGaussianDensity(grid.points, num_s=num_s, num_p=num_p, normalize=True)
    fit = KLDivergenceFPI(grid, density, model, mask_value=1e-18, spherical=True,
                          integral_dens=atomic_numb)

    # Run the Kullback-Leibler FPI Method
    results = fit.run(
        coeffs, e_0, maxiter=10000, c_threshold=1e-6, e_threshold=1e-6, d_threshold=1e-14,
        disp=True
    )
    # Construct Fitting Object using SLSQP and optimizing KL
    # measure = KLDivergence(mask_value=1e-18)
    # fit_KL_slsqp = ScipyFit(grid, density, model, measure=measure, method="SLSQP", spherical=True)
    # # Run the SLSQP optimization algorithm
    # results = fit_KL_slsqp.run(coeffs, e_0, maxiter=10000, disp=True, with_constraint=True, tol=1e-14)

    print("KL-FPI INFO")
    print("-----------")
    print("Success %s" % results["success"])
    print("Final Coeffs")
    print(results["coeffs"])
    print("Final Exponents")
    print(results["exps"])
    print("Integration Value & L1 & L_infinity & LS & KL (With 4 pi r^2 included)")
    p = results["performance"][-1]
    print(p)
    # Calculate the relative errors
    spherical = 4.0 * np.pi * grid.points**2.0
    l1 = results["performance"][-1][1] / grid.integrate(density * spherical)
    linf = results["performance"][-1][2] / np.max(density)
    ls = results["performance"][-1][3] / grid.integrate(density**2.0 * spherical)
    kl = results["performance"][-1][4] / grid.integrate(
        density * np.log(density) * spherical
    )
    print("Relative Errors L1 & L_infinity & LS & KL (With 4 pi r^2 included)")
    print([l1, linf, ls, kl])

    # Store the results
    results_final[element + "_coeffs_s"] = results["coeffs"][:num_s]
    results_final[element + "_exps_s"] = results["exps"][:num_s]
    results_final[element + "_coeffs_p"] = results["coeffs"][num_s:]
    results_final[element + "_exps_p"] = results["exps"][num_s:]
    results_final[element + "_errors_4pir2"] = results["performance"][-1]
    results_final[element + "_sucess"] = results["success"]
    results_final[element + "_errors"] = [l1, linf, ls, kl]

    np.savez("./result_kl_fpi_method_cugbasis_atomdb_hci.npz", **results_final)
