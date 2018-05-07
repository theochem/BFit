import numpy as np
import scipy


def optimize_using_nnls(true_dens, cofactor_matrix):
    b_vector = np.copy(true_dens)
    b_vector = np.ravel(b_vector)
    row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_nnls_valence(true_val_dens, cofactor_matrix):
    b_vector = np.copy(true_val_dens)
    b_vector = np.ravel(b_vector)

    row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_slsqp(density_model, initial_guess, bounds=None, *args):
    if bounds is None:
        bounds = np.array([(0.0, np.inf) for x in range(0, len(initial_guess))], dtype=np.float64)
    f_min_slsqp = scipy.optimize.minimize(density_model.cost_function,
                                          x0=initial_guess,
                                          method="SLSQP",
                                          bounds=bounds, args=(args),
                                          jac=density_model.derivative_of_cost_function)
    parameters = f_min_slsqp['x']
    return parameters


def optimize_using_l_bfgs(density_model, initial_guess, bounds=None, *args):
    if bounds is None:
        bounds = np.array([(0.0, 1.7976931348623157e+308) for x in range(0, len(initial_guess))],
                          dtype=np.float64)
    f_min_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b(density_model.cost_function,
                                                  x0=initial_guess,
                                                  bounds=bounds,
                                                  fprime=density_model.derivative_of_cost_function,
                                                  maxfun=1500000,
                                                  maxiter=1500000,
                                                  factr=1e7,
                                                  args=args, pgtol=1e-5)

    parameters = f_min_l_bfgs_b[0]
    return parameters
