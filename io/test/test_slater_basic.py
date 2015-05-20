import sys
#sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
#sys.path.append("/fitting/")
#import slater_basic as sb
#from fitting import *
import fitting.io.slater_basic as sb
import numpy as np


def test_parsing_slater_density_Be():
    # TODO: put system independet path
    be = sb.load_slater_basis('/Users/Farnaz/code/fitting/fitting/examples/be.txt')

    assert be['configuration'] == '1S(2)2S(2)'
    assert be['energy'] == [-14.573023167, 14.573023130, -29.146046297 ]

    # check basis of S orbitals
    assert be['orbitals'] == ['1S', '2S']
    assert be['orbitals_cusp']['S'] == [1.0001235, 0.9998774]  #turn into array
    assert be['orbitals_energy']['S'] == [-4.7326699, -0.3092695] #turn into array
    #assert len(be['orbitals_basis']) == 1  #uncomment when the structure of the class changes
    assert be['orbitals_basis']['S'] == ['1S', '1S', '1S', '1S', '1S', '1S', '2S', '1S']
    #assert len(be['orbitals_cusp']) == 1  #uncomment when the structure of the class changes
    assert len(be['orbitals_electron_number']) == 2
    assert be['orbitals_electron_number']['1S'] == 2
    assert be['orbitals_electron_number']['2S'] == 2
    #assert be['orbitals_electron_number'] == {'1S': 2, '2S': 2}   #dict is not ordered
    assert be['orbitals_electron_array'].shape == (2, 1)
    assert (be['orbitals_electron_array'] == np.array([[2], [2]])).all()   #rename this orbitals_occupation
    basis_numbers = np.array([[1], [1], [1], [1], [1], [1], [2], [1]])
    assert (be['basis_numbers']['S'] == basis_numbers).all()

    # check exponents of S orbitals
    exponents = np.array([12.683501, 8.105927, 5.152556, 3.472467, 2.349757, 1.406429, 0.821620, 0.786473])
    assert (abs(be['orbitals_exp']['S'] - exponents.reshape(8, 1)) < 1.e-6).all()

    # check coefficients of S orbitals
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562, 0.0315855, -0.0035284, -0.0004149, .0012299])
    assert be['orbitals_coeff']['1S'].shape == (8, 1)
    assert (abs(be['orbitals_coeff']['1S'] - coeff_1s.reshape(8, 1)) < 1.e-6).all()
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910, -0.3598016, -0.2563459, 0.2434108, 1.1150995])
    assert be['orbitals_coeff']['2S'].shape == (8, 1)
    assert (abs(be['orbitals_coeff']['2S'] - coeff_2s.reshape(8, 1)) < 1.e-6).all()


def test_parsing_slater_density_Ag():
    ag = sb.load_slater_basis('/Users/Farnaz/code/fitting/fitting/examples/ag.txt')
    values = ag

    assert values['orbitals'] == ['1S', '2S', '3S', '4S', '5S', '2P', '3P', '4P', '3D', '4D']
    assert values['configuration'] == 'K(2)L(8)M(18)4S(2)4P(6)5S(1)4D(10)'
    assert values['orbitals_basis']['P'] == ['2P', '3P', '3P', '2P', '3P', '3P', '3P', '3P', '2P', '2P', '2P']
    assert values['orbitals_cusp']['P'] == [1.0008130, 1.0008629, 0.9998751]
    assert values['orbitals_cusp']['D'] == [0.9991182, 1.0009214]
    assert values['orbitals_energy']['P'] == [-125.1815809, -21.9454343, -2.6768201]
    assert values['energy'] == [-5197.698467674, 5197.698468984, -10395.396936658]
    assert np.array_equal(   values['orbitals_exp']['D'] ,  np.array(
                                                                    [[53.296212],
                                                                     [40.214567],
                                                                     [21.872645],
                                                                     [17.024065],
                                                                     [10.708021],
                                                                     [7.859216],
                                                                     [5.770205],
                                                                     [3.610289],
                                                                     [ 2.243262],
                                                                     [1.397570],
                                                                     [0.663294]]
                                                                    ))
    assert np.array_equal( np.concatenate((values['orbitals_coeff']['3D'], values['orbitals_coeff']['4D']), axis = 1),
                                           np.array(
                                                    [[ 0.0006646, -0.0002936],
                                                     [0.0037211 , -0.0016839],
                                                     [-0.0072310,  0.0092799],
                                                     [0.1799224 , -0.0743431],
                                                     [0.5205360 , -0.1179494],
                                                     [0.3265622 , -0.2809146],
                                                     [0.0373867 ,  0.1653040],
                                                     [0.0007434 ,  0.4851980],
                                                     [0.0001743 ,  0.4317110],
                                                     [-0.0000474,  0.1737644],
                                                     [0.0000083,   0.0013751]]
                                                    ))
    assert values['orbitals_electron_number'] == {'1S' : 2, '2S' : 2, '3S':2, '4S':2, '5S': 1, '2P': 6, '3P' : 6, '4P':6, '3D' : 10, '4D': 10}
    np.testing.assert_array_almost_equal(values['orbitals_electron_array'], np.array([ [2], [2], [2], [2], [1], [6], [6], [6], [10], [10]]))

#testParsingSilver("/Users/Alireza/Desktop/neutral/ag")

def test_parsing_slater_density_Ne():
    ne = sb.load_slater_basis('/Users/Farnaz/code/fitting/fitting/examples/ne.txt')
    values = ne
    assert values['energy'] == [-128.547098079, 128.547098140, -257.094196219]
    assert values['orbitals_energy']['P'] == [-0.8504095]
    assert values['orbitals_cusp']['P'] == [1.0000509]
    assert values['orbitals_basis']['P'] == ['3P', '2P', '3P', '2P', '2P', '2P', '2P']
    assert values['configuration'] == "1S(2)2S(2)2P(6)"
    assert values['orbitals'] == ['1S', '2S', '2P']
    assert np.array_equal( values['orbitals_exp']['P'], np.array([ [25.731219], [ 10.674843], [ 8.124569], [ 4.295590], [2.648660], [ 1.710436 ], [ 1.304155 ]]) )
    assert np.array_equal( values['orbitals_coeff']['2P'], np.array( [[ 0.0000409], [ 0.0203038], [0.0340866], [0.2801866], [0.3958489], [0.3203928], [ 0.0510413]]))

#testParsingNeon("/Users/Alireza/Desktop/neutral/ne")
