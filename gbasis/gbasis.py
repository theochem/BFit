

import numpy as np

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


class AtomicGaussianBasis():
    '''
    '''
    def __init__(self, contracted_gaussian_basis):
        '''
        '''
        self.cgb = contracted_gaussian_basis


    def types():
        pass
