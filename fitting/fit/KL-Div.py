from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
from fitting.fit.model import Fitting
from scipy.integrate import trapz
import matplotlib.pyplot as plt


ELEMENT_NAME = "Be"
ATOMIC_NUMBER = 4

file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
#Create Grid for the modeling
from fitting.density.radial_grid import *
radial_grid = Radial_Grid(ATOMIC_NUMBER)
NUMBER_OF_CORE_POINTS = 1000; NUMBER_OF_DIFFUSED_PTS = 1000
row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)

fit_be_obj = Fitting(be)

coefficients = fit_be_obj.optimize_using_nnls(be.create_cofactor_matrix(be.UGBS_s_exponents))

GRID = row_grid_points.copy()
EXPONENTS = be.UGBS_s_exponents.copy()
ELECTRON_DENSITY = be.electron_density.copy()
THRESHOLD = 1e-5
#ELECTRON_DENSITY[ELECTRON_DENSITY < THRESHOLD] = THRESHOLD

def update_exponents(COEFFICIENTS, exponents, THRESHOLD=1e-10):
    exponential_matrix = np.exp(-exponents * np.reshape(GRID, (len(GRID), 1))**2)
    gaussian_density = np.dot(exponential_matrix, COEFFICIENTS)

    gaussian_density[gaussian_density < THRESHOLD] = THRESHOLD
    for i in range(0, len(exponents)):
        exponents[i] = 3 / ( 2 * (exponents[i] / np.pi)**1.5 *
                trapz(y=np.ravel(ELECTRON_DENSITY) * np.power(GRID, 2.) * \
                        np.exp(- exponents[i] * np.power(GRID, 2.)) / gaussian_density, x=GRID))

        exponents[exponents < THRESHOLD] = THRESHOLD
    return exponents

def cont_update_exponents(const_coefficients, exponents, number_of_times):
    integration_values = []
    for x in range(0, number_of_times):
        exponents = update_exponents(const_coefficients, exponents)
        model = be.create_model(np.append(const_coefficients, exponents), len(exponents))
        integration_values.append([
            be.integrate_model_using_trapz(model),
            be.measure_error_by_integration_of_difference(be.electron_density, model),
            be.measure_error_by_difference_of_integration(be.electron_density, model)
                                    ])
    return integration_values, exponents

def update_coefficients(coefficients, EXPONENTS, THRESHOLD=1e-10):
    #coefficients[coefficients < THRESHOLD] = 1.
    exponential_matrix = np.exp(-EXPONENTS * np.reshape(GRID, (len(GRID), 1))**2)
    gaussian_density = np.dot(exponential_matrix, coefficients)

    gaussian_density[gaussian_density < THRESHOLD] = THRESHOLD
    for i in range(0, len(coefficients)):
        coefficients[i] = coefficients[i] * (EXPONENTS[i] / np.pi)**1.5 * \
                trapz(y=np.ravel(ELECTRON_DENSITY) * np.exp(- EXPONENTS[i]*np.power(GRID, 2.)) / gaussian_density, x=GRID)

        #coefficients[coefficients < THRESHOLD] = THRESHOLD

        model = be.create_model(np.append(coefficients, EXPONENTS), len(EXPONENTS))
        plt.plot(model, 'b')
        plt.plot(be.electron_density, 'r')
        plt.show()

    return coefficients

def cont_update_coefficients(coefficients, const_exponents, number_of_times):
    integration_values = []
    for x in range(0, number_of_times):
        coefficients = update_coefficients(coefficients, const_exponents)
        model = be.create_model(np.append(coefficients, const_exponents), len(coefficients))
        integration_values.append([
            be.integrate_model_using_trapz(model),
            be.measure_error_by_integration_of_difference(be.electron_density, model),
            be.measure_error_by_difference_of_integration(be.electron_density, model)
                                    ])

    return integration_values, coefficients

coefficients = np.array([np.random.random()*5 + 1 for x in range(0, len(EXPONENTS))])
integration_values, updated_coefficients = cont_update_coefficients(coefficients, EXPONENTS, 100)
import matplotlib.pyplot as plt
plt.plot(np.array(integration_values)[:,0], 'ro', label="Integrate Model")
plt.plot(np.array(integration_values)[:,1], 'go', label="Int(|True - Model|r^2)dr")
plt.plot(np.array(integration_values)[:,2], 'bo', label="|Int(True)r^2dr - Int(Model)r^2dr|")
plt.plot([be.integrated_total_electron_density for x in range(0, len(integration_values))], 'r', label="True Density")
plt.legend()
plt.title("Beryllium - Update Coefficients - UGBS - Coeffs:NNLS - Threshold:1e-10")
#plt.savefig("2016-07-04_KL_Div_Coeffs_NNLS")
plt.show()

exponents = EXPONENTS.copy()
COEFFICIENTS = updated_coefficients.copy()
integration_values_exps, exponents = cont_update_exponents(COEFFICIENTS, exponents, 100)
plt.plot(np.array(integration_values_exps)[:,0], 'ro', label="Integrate Model")
plt.plot(np.array(integration_values_exps)[:,1], 'go', label="Int(|True - Model|r^2)dr")
plt.plot(np.array(integration_values_exps)[:,2], 'bo', label="|Int(True)r^2dr - Int(Model)r^2dr|")
plt.plot([be.integrated_total_electron_density for x in range(0, len(integration_values))], 'r', label="True Density")
plt.legend()
plt.title("Be - Update Exps - Exps: UGBS - Coeffs: Updated Coeffs")
#plt.savefig("2016-07-04_KL_Div_Exps_NNLs_Updated")
#plt.show()


EXPONENTS = be.UGBS_s_exponents.copy()
ALPHA = EXPONENTS[0]
BETA = EXPONENTS[1] / ALPHA
#403990.717648 0.510686183366
def even_tempered_basis_set(exponents, alpha, beta, number_of_basis_sets_to_add):
    for x in range(0, number_of_basis_sets_to_add):
        exponents = np.append(exponents, alpha * beta**(len(exponents)))
    return exponents

EXPONENTS = even_tempered_basis_set(np.array([]), ALPHA, BETA, 30)
coefficients = fit_be_obj.optimize_using_nnls(be.create_cofactor_matrix(EXPONENTS))

#parameters = np.append(coefficients, EXPONENTS)
#parameters = fit_be_obj.optimize_using_l_bfgs(parameters, len(coefficients))
#coefficients = parameters[0:len(coefficients)]
#EXPONENTS = parameters[len(coefficients):]

integration_values_coeff, COEFFICIENTS = cont_update_coefficients(coefficients, EXPONENTS, 100)
integration_values_exp, EXPONENTS = cont_update_exponents(COEFFICIENTS, EXPONENTS, 100)

plt.plot(np.array(integration_values_coeff)[:,0], 'ro', label="Integrate Model")
plt.plot(np.array(integration_values_coeff)[:,1], 'go', label="Int(|True - Model|r^2)dr")
plt.plot(np.array(integration_values_coeff)[:,2], 'bo', label="|Int(True)r^2dr - Int(Model)r^2dr|")
plt.plot([be.integrated_total_electron_density for x in range(0, len(integration_values))], 'r', label="True Density")
plt.title("Be - Update Coeffs -Even-Tempered:100 Exps - Coeffs:NNLS")
plt.legend()
plt.xlabel("Number Of Iterations and Alpha:403990 - Beta:0.5")
#plt.savefig("2016-07-04_KL_Div_Even_tempered_Coeffs")
#plt.show()

plt.plot(np.array(integration_values_exp)[:,0], 'ro', label="Integrate Model")
plt.plot(np.array(integration_values_exp)[:,1], 'go', label="Int(|True - Model|r^2)dr")
plt.plot(np.array(integration_values_exp)[:,2], 'bo', label="|Int(True)r^2dr - Int(Model)r^2dr|")
plt.plot([be.integrated_total_electron_density for x in range(0, len(integration_values))], 'r', label="True Density")
plt.title("Be - Update Exps -Even-Tempered:100 Exps - Coeffs:Updated ")
plt.legend()
plt.xlabel("Number Of Iterations and Alpha:403990 - Beta:0.5")
#plt.savefig("2016-07-04_KL_Div_Even_tempered_Exps")
#plt.show()
