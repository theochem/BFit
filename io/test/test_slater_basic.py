import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
import slater_basic as sb
import numpy as np

def testParsingBeryllium(file):
    values = dict(sb.load_slater_basis(file))

    assert values['configuration'] == "1S(2)2S(2)"
    assert values['energy'] == [-14.573023167 ,   14.573023130, -29.146046297 ]
    assert np.array_equal(values['orbitals_exp']['S'], np.array([[12.683501], [8.105927], [5.152556], [3.472467], [2.349757], [1.406429], [0.821620], [0.786473]]))
    assert np.array_equal(np.concatenate((values['orbitals_coeff']['1S'], values['orbitals_coeff']['2S']), axis = 1), \
                          np.array([[-0.0024917, 0.0004442],
                                    [0.0314015, -0.0030990],
                                    [0.0849694, -0.0367056],
                                    [0.8685562, 0.0138910],
                                    [0.0315855, -0.3598016],
                                    [-0.0035284, -0.2563459],
                                    [-0.0004149, 0.2434108],
                                    [0.0012299, 1.1150995]]))
    assert values['orbitals_basis']['S'] == ['1S', '1S', '1S', '1S', '1S', '1S', '2S', '1S']
    assert values['orbitals'] == ['1S', '2S']
    assert values["orbitals_cusp"] == [1.0001235 ,     0.9998774]
    assert values['orbitals_energy'] == [-4.7326699     ,-0.3092695]

testParsingBeryllium("/Users/Alireza/Desktop/neutral/be")

def testParsingSilver(file):
    values = sb.load_slater_basis(file)

    assert values['orbitals'] == ['1S', '2S', '3S', '4S', '5S' '2P', '3P', '4P', '3D', '4D']

testParsingSilver("/Users/Alireza/Desktop/neutral/ag")