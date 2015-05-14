import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
from atomic_density import *
import slater_basic as sb
import numpy as np
import math

def testSlatorBeryllium(file): #should I put variables?
    assert slator_type_orbital(12.683501, 1, 1) == (2 * 12.683501)**1 * math.sqrt((2 * 12.683501)/ math.factorial(2 * 1)) * 1**0 * math.exp(12.683501 * -1)
    assert slator_type_orbital(0.821620, 2, 2) == (2 * 0.821620) **2 * math.sqrt((2 * 0.821620)/ math.factorial(2 * 2)) * 2**1 * math.exp(-0.821620 * 2)

testSlatorBeryllium("/Users/Alireza/Desktop/neutral/be")

def testLCAOBeryllium(file):
    r = np.array([[1]])
    values = sb.load_slater_basis(file)
    LCAO1S = slator_type_orbital(12.683501, 1, r) * -0.0024917 + slator_type_orbital(18.105927, 1, r) * 0.0314015 + slator_type_orbital(5.152556, 1, r) * 0.0849694 + \
             slator_type_orbital(3.472467, 1, r) * 0.8685562 + slator_type_orbital(2.349757, 1, r) * 0.0315855 + slator_type_orbital(1.406429, 1, r) * -0.0035284 + \
             slator_type_orbital(0.821620, 2, r) * -0.0004149 + slator_type_orbital(0.786473, 1, r) * 0.0012299
    LCAO2S = slator_type_orbital(12.683501, 1, r) * 0.0004442 + slator_type_orbital(18.105927, 1, r) * -0.0030990 + slator_type_orbital(5.152556, 1, r) * -0.0367056 + \
             slator_type_orbital(3.472467, 1, r) * 0.0138910 + slator_type_orbital(2.349757, 1, r) * -0.3598016 + slator_type_orbital(1.406429, 1, r) * -0.2563459 + \
             slator_type_orbital(0.821620, 2, r) * 0.2434108 + slator_type_orbital(0.786473, 1, r) * 1.1150995
    LCAOFunc1S = phi_LCAO((slator_type_orbital(values['orbitals_exp']['S'], sb.getQuantumNumber(open(file, 'r').read(), 'S'), r )) ,values['orbitals_coeff']['1S'])
    LCAOFunc2S =  phi_LCAO((slator_type_orbital(values['orbitals_exp']['S'], sb.getQuantumNumber(open(file, 'r').read(), 'S'), r )) ,values['orbitals_coeff']['2S'])

    assert np.absolute(LCAO1S - LCAOFunc1S[0, 0]) < 0.001
    assert np.absolute(LCAO2S - LCAOFunc2S[0 ,0]) < 0.0001

testLCAOBeryllium("/Users/Alireza/Desktop/neutral/be")