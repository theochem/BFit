
import os
from fitting.gbasis.gbasis_nwchem import load_gbasis_nwchem_format, fortran_float
import numpy as np


def test_load_basis_nwchem_ugbs_h():
    # Test the ugbs basis set for all atoms
    file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/basis/ugbs.nwchem'
    basis_atom_map = load_gbasis_nwchem_format(file_path)
    # check that all the 86 elements (H to Rn) are read in
    assert len(basis_atom_map) == 72
    # ugbs basis for H consist of 20 contracted gaussian basis
    basis_h = basis_atom_map['h']
    assert len(basis_h) == 20
    # check that all contracted gaussian basis are s type and have length one
    for basis in basis_h:
        assert basis.__length__() == 1
        assert basis.shell_type == 's'
        assert abs(basis.coefficients - np.array([1])) < 1.e-6
    # check exponent of a couple of contracted gaussian basis
    assert abs(basis_h[0].exponents - np.array([13739.0854166734])) < 1.e-6
    assert abs(basis_h[2].exponents - np.array([3583.15866840946])) < 1.e-6
    assert abs(basis_h[6].exponents - np.array([243.715119463182])) < 1.e-6
    assert abs(basis_h[10].exponents - np.array([16.5767315800501])) < 1.e-6
    assert abs(basis_h[15].exponents - np.array([0.575797063890464])) < 1.e-6
    assert abs(basis_h[19].exponents - np.array([3.916389509898707e-002])) < 1.e-6


def test_load_basis_nwchem_ugbs_be():
    # Test the ugbs basis set for all atoms
    file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/basis/ugbs.nwchem'
    basis_atom_map = load_gbasis_nwchem_format(file_path)
    # check that all the 86 elements (H to Rn) are read in
    assert len(basis_atom_map) == 72
    # ugbs basis for Be consist of 25 contracted gaussian basis
    basis_be = basis_atom_map['be']
    assert len(basis_be) == 25
    # check that all contracted gaussian basis are s type and have length one
    for basis in basis_be:
        assert basis.__length__() == 1
        assert basis.shell_type == 's'
        assert abs(basis.coefficients - np.array([1])) < 1.e-6
    # check exponent of a couple of contracted gaussian basis
    assert abs(basis_be[0].exponents - np.array([201995.358823884])) < 1.e-6
    assert abs(basis_be[3].exponents - np.array([26903.1860742975])) < 1.e-6
    assert abs(basis_be[7].exponents - np.array([1829.86962476550])) < 1.e-6
    assert abs(basis_be[13].exponents - np.array([32.4597220758638])) < 1.e-6
    assert abs(basis_be[21].exponents - np.array([0.150168091845475])) < 1.e-6
    assert abs(basis_be[24].exponents - np.array([2.000046011385546e-002])) < 1.e-6


def test_load_basis_nwchem_ugbs_ne():
    # Test the ugbs basis set for all atoms
    file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/basis/ugbs.nwchem'
    basis_atom_map = load_gbasis_nwchem_format(file_path)
    # check that all the 86 elements (H to Rn) are read in
    assert len(basis_atom_map) == 72
    # ugbs basis for Ne consist of 23 type-s and 16 type-p contracted gaussian basis
    basis_ne = basis_atom_map['ne']
    assert len(basis_ne) == 23 + 16
    # check that all contracted gaussian basis are s or p type and have length one
    type_s, type_p = 0, 0
    for basis in basis_ne:
        assert basis.__length__() == 1
        assert abs(basis.coefficients - np.array([1])) < 1.e-6
        if basis.shell_type == 's':
            type_s += 1
        elif basis.shell_type == 'p':
            type_p +=1
        else:
            raise ValueError('Ne should only have only s and p type contracted gaussian basis. error={0}'.format(basis.shell_type))
    assert type_s == 23
    assert type_p == 16
    # check exponent of a couple of contracted gaussian basis
    assert abs(basis_ne[0].exponents - np.array([395537.152566831])) < 1.e-6
    assert abs(basis_ne[5].exponents - np.array([13739.0854166734])) < 1.e-6
    assert abs(basis_ne[12].exponents - np.array([124.461944187287])) < 1.e-6
    assert abs(basis_ne[18].exponents - np.array([2.20780762884063])) < 1.e-6
    assert abs(basis_ne[24].exponents - np.array([934.489134729211])) < 1.e-6
    assert abs(basis_ne[27].exponents - np.array([124.461944187287])) < 1.e-6
    assert abs(basis_ne[31].exponents - np.array([8.46550778330153])) < 1.e-6
    assert abs(basis_ne[33].exponents - np.array([2.20780762884063])) < 1.e-6
    assert abs(basis_ne[38].exponents - np.array([7.668876968794860e-002])) < 1.e-6


def test_load_basis_nwchem_ugbs_tc():
    # Test the ugbs basis set for all atoms
    file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/basis/ugbs.nwchem'
    basis_atom_map = load_gbasis_nwchem_format(file_path)
    # check that all the 86 elements (H to Rn) are read in
    assert len(basis_atom_map) == 72
    # ugbs basis for Tc consist of 32 type-s, 22 type-p, and 17 type-d contracted gaussian basis
    basis_tc = basis_atom_map['tc']
    assert len(basis_tc) == 32 + 22 + 17
    # check that all contracted gaussian basis are s, p or d type and have length one
    type_s, type_p, type_d = 0, 0, 0
    for basis in basis_tc:
        assert basis.__length__() == 1
        assert abs(basis.coefficients - np.array([1])) < 1.e-6
        if basis.shell_type == 's':
            type_s += 1
        elif basis.shell_type == 'p':
            type_p +=1
        elif basis.shell_type == 'd':
            type_d +=1
        else:
            raise ValueError('Tc should only have only s and p type contracted gaussian basis. error={0}'.format(basis.shell_type))
    assert type_s == 32
    assert type_p == 22
    assert type_d == 17
    # check exponent of a couple of contracted gaussian basis
    assert abs(basis_tc[0].exponents - np.array([22297831.7330222])) < 1.e-6
    assert abs(basis_tc[35].exponents - np.array([13739.0854166734])) < 1.e-6
    assert abs(basis_tc[70].exponents - np.array([7.668876968794860e-002])) < 1.e-6
