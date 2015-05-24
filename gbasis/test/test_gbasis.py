
import numpy as np
from fitting.gbasis.gbasis import *


def test_ContractedGaussianBasis():
    cgb = ContractedGaussianBasis('s', [1, 2, 3.0], [4, 5, 6])
    cgb.to_arrays()
    assert (abs(cgb.exponents - np.array([1.0, 2.0, 3.0])) < 1.e-6).all()
    assert (abs(cgb.coefficients - np.array([4, 5, 6])) < 1.e-6).all()
    assert cgb.__length__() == 3
