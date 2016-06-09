from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *



def norm(x_values, center):
    norm_value = 0

    for i in range(0, len(x_values)):
        norm_value += (x_values[i] - center[i])**2
    return norm_value

def gaussian(beta, x_values, center):
    return np.exp(-beta * norm(x_values, center))

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)

    number_of_points = (be.electron_density.shape[0])



    G = np.empty((number_of_points, number_of_points))
    for row_index in range(0,number_of_points ):
        for col_index in range(0, number_of_points):
            G[row_index, col_index] = gaussian(100000., [row_grid_points[col_index]], [row_grid_points[row_index]])

    weights = scipy.optimize.nnls(G, np.ravel(be.electron_density))[0]

    print(weights)

    result = np.dot(G, weights)
    print(result.shape)
    print(np.sum((result > 0)))
    inte = np.trapz(y=np.ravel(column_grid_points**2) * np.ravel(result), x=np.ravel(np.ravel(column_grid_points)))
    print(inte)
    print(be.integrated_total_electron_density)
    plt.plot(result)
    plt.plot(be.electron_density)
    plt.show()





