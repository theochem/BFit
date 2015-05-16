from fitting2 import *
import numpy as np



def testBeryllium(beFile):
    assert np.array_equal( getExponents(beFile, "S"), np.array([[12.683501], [8.105927], [5.152556], [3.472467], [2.349757], [1.406429], [0.821620], [0.786473]]))

    assert np.array_equal( getQuantumNumber(beFile, "S"), np.array([[1],[1],[1],[1],[1],[1],[2],[1]] ))

    assert np.array_equal(getCoefficients(beFile, 'S'), np.array([[-0.0024917, 0.0004442],
                                                             [0.0314015, -0.0030990],
                                                             [0.0849694, -0.0367056],
                                                             [0.8685562, 0.0138910],
                                                             [0.0315855, -0.3598016],
                                                             [-0.0035284, -0.2563459],
                                                             [-0.0004149, 0.2434108],
                                                             [0.0012299, 1.1150995]]))


    assert sum(getOccupationNumber(beFile).values()) == 4 #sum of electrons equal to 4
    assert getOccupationNumber(beFile).get("1S") == 2 and getOccupationNumber(beFile).get("2S") == 2

    assert getSlatorTypeOrbital(12.683501, 1, 1) == (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(12.683501 * -1)
    assert getSlatorTypeOrbital(0.821620, 2, 2) == (2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)

    r = np.array([[1]])
    LCAO1S = getSlatorTypeOrbital(12.683501, 1, r) * -0.0024917 + getSlatorTypeOrbital(8.105927, 1, r) * 0.0314015 + getSlatorTypeOrbital(5.152556, 1, r) * 0.0849694 + \
             getSlatorTypeOrbital(3.472467, 1, r) * 0.8685562 + getSlatorTypeOrbital(2.349757, 1, r) * 0.0315855 + getSlatorTypeOrbital(1.406429, 1, r) * -0.0035284 + \
             getSlatorTypeOrbital(0.821620, 2, r) * -0.0004149 + getSlatorTypeOrbital(0.786473, 1, r) * 0.0012299

    LCAO2S = getSlatorTypeOrbital(12.683501, 1, r) * 0.0004442 + getSlatorTypeOrbital(8.105927, 1, r) * -0.0030990 + getSlatorTypeOrbital(5.152556, 1, r) * -0.0367056 + \
             getSlatorTypeOrbital(3.472467, 1, r) * 0.0138910 + getSlatorTypeOrbital(2.349757, 1, r) * -0.3598016 + getSlatorTypeOrbital(1.406429, 1, r) * -0.2563459 + \
             getSlatorTypeOrbital(0.821620, 2, r) * 0.2434108 + getSlatorTypeOrbital(0.786473, 1, r) * 1.1150995
    LCAOFunc = getLCAO((getSlatorTypeOrbital(getExponents(beFile, 'S'), getQuantumNumber(beFile, 'S'), r )) ,getCoefficients(beFile, "S"))

    assert np.absolute(LCAO1S - LCAOFunc[0,0]) < 0.000000000001 and np.absolute(LCAO2S - LCAOFunc[0,1]) < 0.00000000001 #different error values, ask Farnaz

    #assert np.absolute( rho(getOccupationNumber(beFile), LCAO1S)  - LCAOFunc[0,0] * 2) < 0.001
    #assert np.absolute( rho(getOccupationNumber(beFile), LCAO2S) - LCAOFunc[0, 1] * 2) < 0.001


testBeryllium("/Users/Alireza/Desktop/neutral/be")

def testKrypton(krFile):
    assert np.array_equal( getExponents(krFile, "P"), np.array([ [55.423460], [45.348388], [26.746410], [21.299661], [15.365217], [8.075067], [6.762676], [4.956675], [1.641161], [1.395473], [1.023901]]))

    assert np.array_equal( getQuantumNumber(krFile, 'P'), np.array([[2], [3], [2], [3], [3], [2], [3], [3], [3], [2], [2]]))

    assert np.array_equal( getCoefficients(krFile, 'P'), np.array([[-0.0062132,     -0.0039511,     -0.0012127],
                                                                   [-0.0133944,     -0.0084081,     -0.0025837],
                                                                   [0.3355068,      0.1656088,      0.0478133],
                                                                   [ 0.4219443,      0.1992198,      0.0558041],
                                                                   [0.3213774,     0.3102247,      0.1007406],
                                                                   [0.0286895,     -0.3970577,     -0.1665597],
                                                                   [-0.0034907,     -0.6549448,     -0.1758964],
                                                                   [0.0012162,     -0.2207516,     -0.3108946],
                                                                   [ 0.0003760,      0.0056800,     -0.5032745],
                                                                   [-0.0004632,     -0.0090577,      1.5227627],
                                                                   [-0.0000170,      0.0004744,      0.1439485]]))

    assert sum(getOccupationNumber(krFile).values()) == 36
    assert getOccupationNumber(krFile).get("2P") == getOccupationNumber(krFile).get("3P") == getOccupationNumber(krFile).get("4P")
testKrypton("/Users/Alireza/Desktop/neutral/kr")

def testSilver(agFile):
    assert np.array_equal( getExponents(agFile, "D"), np.array([ [53.296212], [40.214567 ], [21.872645], [17.024065], [10.708021], [7.859216], [5.770205], [3.610289], [ 2.243262], [1.397570], [0.663294]]))

    assert np.array_equal( getQuantumNumber(agFile, 'D'), np.array([ [3], [4], [3], [3], [3] ,[3] ,[3] ,[3] ,[3] ,[3], [3]]))

    assert np.array_equal( getCoefficients(agFile, 'D'), np.array([[ 0.0006646,     -0.0002936],
                                                                   [0.0037211 ,    -0.0016839],
                                                                   [-0.0072310,      0.0092799],
                                                                   [0.1799224 ,    -0.0743431],
                                                                   [0.5205360 ,    -0.1179494],
                                                                   [0.3265622 ,    -0.2809146],
                                                                   [0.0373867 ,     0.1653040],
                                                                   [0.0007434 ,     0.4851980],
                                                                   [0.0001743 ,     0.4317110],
                                                                   [-0.0000474,      0.1737644],
                                                                   [0.0000083,     0.0013751]]))


    assert sum(getOccupationNumber(agFile).values()) == 47
    assert getOccupationNumber(agFile).get("3D") == getOccupationNumber(agFile).get("4D") == 10
    assert getOccupationNumber(agFile).get("5S") == 1
testSilver("/Users/Alireza/Desktop/neutral/ag")

