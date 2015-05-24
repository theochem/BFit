import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
from atomic_density import *
import slater_basic as sb
import numpy as np
import math


def testBeryllium(file):
    be = Atomic_Density(file, np.array([[1]]))

    def testSlatorBeryllium():
        # Test Values Of The Slator Type Orbital Equation
        assert be.slator_type_orbital(12.683501, 1, 1) == (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(-12.683501 * 1)
        assert be.slator_type_orbital(0.821620, 2, 2) == (2 * 0.821620)**2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)

        # Test Values In the form of an array of the Slator Type Orbital Equation.
        Exp_Array = np.array([[12.683501], [0.821620]])
        Quantum_Array = np.array([ [1], [2]])
        #the grid cannot be in vertical form
        grid2 = np.array([ [1], [2]])
        grid = [1, 2]
        # rows are the slator_Type orbital, where each column represents each point in the grid
        np.testing.assert_array_almost_equal(  be.slator_type_orbital(Exp_Array, Quantum_Array, grid) ,
                np.array([ [(2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(-12.683501 * 1), (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 2**0 * math.exp(-12.683501 * 2) ],
                           [(2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 1**1 * math.exp(-0.821620 * 1) , (2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)]]))

    testSlatorBeryllium()


    def testLCAOBeryllium():

        #Test One Value
        r = np.array([[1]])
        LCAO1S = be.slator_type_orbital(12.683501, 1, r)*-0.0024917 + be.slator_type_orbital(8.105927, 1, r)*0.0314015 + be.slator_type_orbital(5.152556, 1, r)*0.0849694 + \
                 be.slator_type_orbital(3.472467, 1, r)*0.8685562 + be.slator_type_orbital(2.349757, 1, r)*0.0315855 + be.slator_type_orbital(1.406429, 1, r)*-0.0035284 + \
                 be.slator_type_orbital(0.821620, 2, r)*-0.0004149 + be.slator_type_orbital(0.786473, 1, r)*0.0012299
        LCAO2S = be.slator_type_orbital(12.683501, 1, r)*0.0004442 + be.slator_type_orbital(8.105927, 1, r)*-0.0030990 + be.slator_type_orbital(5.152556, 1, r)*-0.0367056 + \
                 be.slator_type_orbital(3.472467, 1, r)*0.0138910 + be.slator_type_orbital(2.349757, 1, r)*-0.3598016 + be.slator_type_orbital(1.406429, 1, r)*-0.2563459 + \
                 be.slator_type_orbital(0.821620, 2, r)*0.2434108 + be.slator_type_orbital(0.786473, 1, r)*1.1150995
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
                                                                                     [0.0012299,      1.1150995]]
                                                                                   ))

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

#testBeryllium("/Users/Alireza/Desktop/neutral/be")

def test_silver(file):
    ag = Atomic_Density(file, np.array([[1]]))

    def test_slator_silver():
        assert ag.slator_type_orbital(58.031368, 2, 1) == (2 * 58.031368)**2 * math.sqrt((2 * 58.031368) / math.factorial(2 * 2)) * 1**(2 - 1) * math.exp(58.031368*-1*1)
        assert ag.slator_type_orbital(51.182468, 3, 1) == (2 * 51.182468)**3 * math.sqrt((2 * 51.182468)/math.factorial(2 * 3)) * 1**(3-1) * math.exp(51.182468*-1)

    test_slator_silver()

    def test_LCAO_silver():
        LCAO3D = ag.slator_type_orbital(53.296212, 3, 1) * 0.0006646 + ag.slator_type_orbital(40.214567, 4, 1) * 0.0037211 + ag.slator_type_orbital(21.872645, 3, 1) \
                 * -0.0072310 + ag.slator_type_orbital(17.024065, 3, 1)  *  0.1799224 + ag.slator_type_orbital(10.708021, 3, 1) * 0.5205360 + ag.slator_type_orbital(7.859216 , 3, 1) \
                 * 0.3265622 + ag.slator_type_orbital(5.770205, 3, 1) * 0.0373867 + ag.slator_type_orbital(3.610289, 3, 1) * 0.0007434 + ag.slator_type_orbital(2.243262 , 3, 1) \
                 *  0.0001743 + ag.slator_type_orbital(1.397570 , 3, 1) * -0.0000474 + ag.slator_type_orbital(0.663294, 3, 1) * 0.0000083

        LCAO4D =  ag.slator_type_orbital(53.296212, 3, 1) * -0.0002936 + ag.slator_type_orbital(40.214567, 4, 1) * -0.0016839 + ag.slator_type_orbital(21.872645, 3, 1) \
                 * 0.0092799 + ag.slator_type_orbital(17.024065, 3, 1)  *  -0.0743431 + ag.slator_type_orbital(10.708021, 3, 1) * -0.1179494 + ag.slator_type_orbital(7.859216 , 3, 1) \
                 * -0.2809146 + ag.slator_type_orbital(5.770205, 3, 1) *0.1653040 + ag.slator_type_orbital(3.610289, 3, 1) *  0.4851980 + ag.slator_type_orbital(2.243262 , 3, 1) \
                 *  0.4317110 + ag.slator_type_orbital(1.397570 , 3, 1) * 0.1737644 + ag.slator_type_orbital(0.663294, 3, 1) * 0.0013751

        LCAOFunc = ag.phi_LCAO('D')

        np.testing.assert_almost_equal(LCAO3D , LCAOFunc[0, 0], decimal=16)
        np.testing.assert_almost_equal(LCAO4D, LCAOFunc[0, 1], decimal = 15)
    test_LCAO_silver()

    def test_all_coeff_matrix():
                                                                                    #3D          4D
        np.testing.assert_array_almost_equal(ag.all_coeff_matrix('D'), np.array([ [0.0006646, -0.0002936],
                                                                                  [0.0037211,  -0.0016839],
                                                                                  [-0.0072310, 0.0092799],
                                                                                  [0.1799224, -0.0743431],
                                                                                  [0.5205360, -0.1179494],
                                                                                  [0.3265622, -0.2809146],
                                                                                  [0.0373867, 0.1653040],
                                                                                  [0.0007434, 0.4851980],
                                                                                  [0.0001743, 0.4317110],
                                                                                  [-0.0000474, 0.1737644],
                                                                                  [0.0000083, 0.0013751]]
                                                                                ))
    test_all_coeff_matrix()

    def test_phi_matrix():
        print(ag.phi_matrix())
        phi3D = ag.slator_type_orbital(53.296212, 3, 1) * 0.0006646 + ag.slator_type_orbital(40.214567, 4, 1) * 0.0037211 + ag.slator_type_orbital(21.872645, 3, 1) \
                 * -0.0072310 + ag.slator_type_orbital(17.024065, 3, 1)  *  0.1799224 + ag.slator_type_orbital(10.708021, 3, 1) * 0.5205360 + ag.slator_type_orbital(7.859216 , 3, 1) \
                 * 0.3265622 + ag.slator_type_orbital(5.770205, 3, 1) * 0.0373867 + ag.slator_type_orbital(3.610289, 3, 1) * 0.0007434 + ag.slator_type_orbital(2.243262 , 3, 1) \
                 *  0.0001743 + ag.slator_type_orbital(1.397570 , 3, 1) * -0.0000474 + ag.slator_type_orbital(0.663294, 3, 1) * 0.0000083

        phi4D = ag.slator_type_orbital(53.296212, 3, 1) * -0.0002936 + ag.slator_type_orbital(40.214567, 4, 1) * -0.0016839 + ag.slator_type_orbital(21.872645, 3, 1) \
                 * 0.0092799 + ag.slator_type_orbital(17.024065, 3, 1)  *  -0.0743431 + ag.slator_type_orbital(10.708021, 3, 1) * -0.1179494 + ag.slator_type_orbital(7.859216 , 3, 1) \
                 * -0.2809146 + ag.slator_type_orbital(5.770205, 3, 1) *0.1653040 + ag.slator_type_orbital(3.610289, 3, 1) *  0.4851980 + ag.slator_type_orbital(2.243262 , 3, 1) \
                 *  0.4317110 + ag.slator_type_orbital(1.397570 , 3, 1) * 0.1737644 + ag.slator_type_orbital(0.663294, 3, 1) * 0.0013751


        np.testing.assert_array_almost_equal(ag.phi_matrix()[0][8:], [phi3D] + [phi4D], decimal = 16)

    test_phi_matrix()
test_silver("/Users/Alireza/Desktop/neutral/ag")
