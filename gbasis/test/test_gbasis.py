
import numpy as np
from fitting.gbasis.gbasis import *


def test_ContractedGaussianBasis():
    cgb = ContractedGaussianBasis('s', [1, 2, 3.0], [4, 5, 6])
    cgb.to_arrays()
    assert (abs(cgb.exponents - np.array([1.0, 2.0, 3.0])) < 1.e-6).all()
    assert (abs(cgb.coefficients - np.array([4, 5, 6])) < 1.e-6).all()
    assert cgb.__length__() == 3


def test_UGBSBasis_be():
    be = UGBSBasis('Be')
    assert len(be.basis) == 25
    assert be.shell_types_count() == [('s', 25)]
    expected = np.array([201995.358823884, 103156.238855453, 52680.4659115021, 26903.1860742975, 13739.0854166734, 7016.36109438299, \
                         3583.15866840945, 1829.86962476550, 934.489134729211, 477.230689612032, 243.715119463182, 124.461944187287, \
                         63.5609952513411, 32.4597220758638, 16.5767315800501, 8.46550778330153, 4.32321786011102, 2.20780762884063, \
                         1.12749685157938, 0.57579706389046, 0.29405160495168, 0.15016809184547, 7.66887696879e-2, 3.91638950989e-2, \
                         2.00004601138e-2])
    assert (abs(be.exponents('s') - expected) < 1.e-6).all()
    assert (abs(be.exponents('S') - expected) < 1.e-6).all()


def test_UGBSBasis_ne():
    ne = UGBSBasis('Ne')
    assert len(ne.basis) == 23 + 16
    types = ne.shell_types_count()
    assert len(types) == 2
    assert ('s', 23) in types
    assert ('p', 16) in types
    exponents_s = ne.exponents('S')
    assert exponents_s.shape == (23,)
    assert abs(exponents_s[0] - 395537.152566831) < 1.e-6
    assert abs(exponents_s[7] - 3583.15866840945) < 1.e-6
    assert abs(exponents_s[14] - 32.4597220758638) < 1.e-6
    assert abs(exponents_s[17] - 4.32321786011102) < 1.e-6
    assert abs(exponents_s[22] - 0.150168091845475) < 1.e-6
    exponents_p = ne.exponents('p')
    assert exponents_p.shape == (16,)
    assert abs(exponents_p[0] - 1829.86962476550) < 1.e-6
    assert abs(exponents_p[6] - 32.4597220758638) < 1.e-6
    assert abs(exponents_p[9] - 4.32321786011102) < 1.e-6
    assert abs(exponents_p[13] - 0.294051604951678) < 1.e-6
    assert abs(exponents_p[15] - 7.668876968794e-2) < 1.e-6
