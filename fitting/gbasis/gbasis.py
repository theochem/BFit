

import os
import numpy as np

__all__ = ["ContractedGaussianBasis"]


class ContractedGaussianBasis():
    ''' A Contracted Gaussian Basis represented by a linear combination of gaussian basis primitives of the same type.
    '''
    def __init__(self, shell_type, exponents, coefficients):
        '''
            **Arguments:**

            shell_type
                A string of length one representing the type (quantum angular number) of the contracted gaussian basis.
                options: 's', 'p', 'd', 'f', 'g', 'h', 'i'

            exponents
                A 1D array containing the exponents of the primitives included in the contracted gaussian basis.

            coefficients
                A 1D array containing the coefficients of the primitives included in the contracted gaussian basis.

            It is possible to construct this object with lists instead of arrays. Just call to_arrays() method once the
            lists are completed. (This is convinient when loading a basis set from a file)
        '''
        if not isinstance(shell_type, str):
            raise ValueError('shell_type should be a string. type(sell_type)={0}'.format(type(shell_type)))
        if len(shell_type) != 1:
            raise ValueError('shell_type={0} should be a string of length one.'.format(shell_type))
        if len(exponents) != len(coefficients):
            raise ValueError('exponents and coefficients should have the same length. {0}!={1}'.format(len(exponents), len(coefficients)))
        # TODO: Check that the type of exponents and coefficients is either list or array
        self.shell_type = shell_type.lower()
        self.exponents = exponents
        self.coefficients = coefficients

    def to_arrays(self):
        '''
        '''
        self.exponents = np.array(self.exponents)
        self.coefficients = np.array(self.coefficients)

    def __length__(self):
        '''
        '''
        assert len(self.exponents) == len(self.coefficients)
        return len(self.exponents)


class UGBSBasis():
    ''' UGBS basis set for an atom. This class is created because we are going to use
        UGBS basis for fitting.

    '''
    def __init__(self, element):
        '''
            **Arguments:**

            _element
                A string representing the _element for which the UGBS basis set is constructed.

        '''
        from gbasis.gbasis_nwchem import load_gbasis_nwchem_format

        # Loading the ugbs basis set
        file_path = os.path.dirname(__file__).rsplit('/', 1)[0][:-7] + \
                    '/fitting/data/basis/ugbs.nwchem'
        #file_path = os.getcwd()[:-4]  + '/data/basis/ugbs.nwchem'
        #file_path = "/work/tehrana/fitting/fitting/data/basis/ugbs.nwchem"
        basis_atom_map = load_gbasis_nwchem_format(file_path)
        # Make sure the given _element exist in the loaded basis set
        if element.lower() not in basis_atom_map.keys():
            raise ValueError('_element={0} does not exist in the basis set loaded from {1}'.format(element, file_path))
        self.element = element.lower()
        self.basis = basis_atom_map[self.element]
        # Make sure that all contracted gaussian basis consist of only one primitive with coefficient one.
        for cgb in self.basis:
            assert cgb.__length__() == 1
            assert abs(cgb.coefficients - np.array([1.0])) < 1.e-6


    def shell_types_count(self):
        ''' Return a list containing the types of the basis shells and the number of each shell.
        '''
        record = []
        for cgb in self.basis:
            record.append(cgb.shell_type)
        result = [(shell, record.count(shell)) for shell in set(record)]
        return result

    def exponents(self, shell_type='s'):
        ''' Return an array containing the primitive exponents of all basis of the specified type.
        '''
        if not isinstance(shell_type, str) and len(shell_type_type) != 1:
            raise ValueError('shell_type is a string of length one. error={0}'.format(shell_type))
        result = []
        for cgb in self.basis:
            if cgb.shell_type == shell_type.lower():
                result.append(cgb.exponents[0])
        result = np.array(result)
        return result
