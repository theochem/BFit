


class Molecular_Density_Construction():
    def __init__(self, dict_of_elements, grid):
        pass


    def construct_electron_density(self):
        pass




if __name__ == "__main__":
    ELEMENT = "BE"
    ATOMIC_NUMBER = 4
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT + ".slater"

    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))




