import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\fitting\io')
from atomic_density import *
import slater_basic as sb
import numpy as np
import math


def testBeryllium(file):
    be = Electron_Structure(file, np.array([[1]]))


    def testSlatorBeryllium():


        # Test If Values Are The Same From Manual
        assert be.slator_type_orbital(12.683501, 1, 1) == (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(-12.683501 * 1)
        assert be.slator_type_orbital(0.821620, 2, 2) == (2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)

        # Test If Values are in the form of an array
        Exp_Array = np.array([[12.683501], [0.821620]])
        Quantum_Array = np.array([ [1], [2]])
        #the grid cannot be in vertical form
        grid2 = np.array([ [1], [2]])
        grid = [1, 2]
        # rows are the slator_Type orbital, where each column represents each point in the grid
        np.testing.assert_array_almost_equal(be.slator_type_orbital(Exp_Array, Quantum_Array, grid) ,
                np.array([ [(2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(-12.683501 * 1), (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 2**0 * math.exp(-12.683501 * 2) ],
                           [(2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 1**1 * math.exp(-0.821620 * 1) , (2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)]]))

    testSlatorBeryllium()


    def testLCAOBeryllium():

        #Test One Value
        r = np.array([[1]])
        LCAO1S = be.slator_type_orbital(12.683501, 1, r) * -0.0024917 + be.slator_type_orbital(8.105927, 1, r) * 0.0314015 + be.slator_type_orbital(5.152556, 1, r) * 0.0849694 + \
                 be.slator_type_orbital(3.472467, 1, r) * 0.8685562 + be.slator_type_orbital(2.349757, 1, r) * 0.0315855 + be.slator_type_orbital(1.406429, 1, r) * -0.0035284 + \
                 be.slator_type_orbital(0.821620, 2, r) * -0.0004149 + be.slator_type_orbital(0.786473, 1, r) * 0.0012299
        LCAO2S = be.slator_type_orbital(12.683501, 1, r) * 0.0004442 + be.slator_type_orbital(8.105927, 1, r) * -0.0030990 + be.slator_type_orbital(5.152556, 1, r) * -0.0367056 + \
                 be.slator_type_orbital(3.472467, 1, r) * 0.0138910 + be.slator_type_orbital(2.349757, 1, r) * -0.3598016 + be.slator_type_orbital(1.406429, 1, r) * -0.2563459 + \
                 be.slator_type_orbital(0.821620, 2, r) * 0.2434108 + be.slator_type_orbital(0.786473, 1, r) * 1.1150995
        LCAOFunc = be.phi_LCAO('S')

        assert math.fabs(LCAO1S[0, 0] - LCAOFunc[0, 0]) < 0.000000000000001
        assert math.fabs(LCAO2S[0, 0] - LCAOFunc[0, 1]) < 0.0000000000000001

    testLCAOBeryllium()


    def test_all_coeff_matrix():                                                     # 1S                 2S
         np.testing.assert_array_almost_equal(be.all_coeff_matrix('S') ,  np.array([ [-0.0024917,      0.0004442],
                                                                                     [0.0314015,     -0.0030990],
                                                                                     [ 0.0849694,     -0.0367056],
                                                                                     [ 0.8685562,      0.0138910],
                                                                                     [0.0315855,     -0.3598016],
                                                                                     [-0.0035284,     -0.2563459],
                                                                                     [-0.0004149,      0.2434108],
                                                                                     [0.0012299,      1.1150995]]))

    test_all_coeff_matrix()


    def test_phi_matrix():
        phi_matrix = be.phi_matrix()
        r = np.array([[1]])

        phi1S = be.slator_type_orbital(12.683501, 1, r) * -0.0024917 + be.slator_type_orbital(8.105927, 1, r) * 0.0314015 + be.slator_type_orbital(5.152556, 1, r) * 0.0849694 + \
                 be.slator_type_orbital(3.472467, 1, r) * 0.8685562 + be.slator_type_orbital(2.349757, 1, r) * 0.0315855 + be.slator_type_orbital(1.406429, 1, r) * -0.0035284 + \
                 be.slator_type_orbital(0.821620, 2, r) * -0.0004149 + be.slator_type_orbital(0.786473, 1, r) * 0.0012299

        phi2S = be.slator_type_orbital(12.683501, 1, r) * 0.0004442 + be.slator_type_orbital(8.105927, 1, r) * -0.0030990 + be.slator_type_orbital(5.152556, 1, r) * -0.0367056 + \
                 be.slator_type_orbital(3.472467, 1, r) * 0.0138910 + be.slator_type_orbital(2.349757, 1, r) * -0.3598016 + be.slator_type_orbital(1.406429, 1, r) * -0.2563459 + \
                 be.slator_type_orbital(0.821620, 2, r) * 0.2434108 + be.slator_type_orbital(0.786473, 1, r) * 1.1150995

        np.testing.assert_array_almost_equal(phi_matrix, np.concatenate((phi1S, phi2S), axis = 1), decimal = 14)


    test_phi_matrix()


    def test_atomic_density():
        atomic_density = be.atomic_density()
        np.testing.assert_array_almost_equal(atomic_density,     np.array([[ (  (be.phi_matrix()[0, 0]**2) * 2  + (be.phi_matrix()[0, 1]**2) * 2)]])      )

    test_atomic_density()


testBeryllium("/Users/Alireza/Desktop/neutral/be")


