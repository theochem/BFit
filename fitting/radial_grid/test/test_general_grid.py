# -*- coding: utf-8 -*-
# FittingBasisSets is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The FittingBasisSets Development Team.
#
# This file is part of FittingBasisSets.
#
# FittingBasisSets is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# FittingBasisSets is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Test file for 'fitting.radial_grid.general_grid'."""


import numpy as np
import numpy.testing as npt
from fitting.radial_grid.general_grid import RadialGrid


def test_input_for_radial_grid():
    r"""Input checks for general grid."""
    npt.assert_raises(TypeError, RadialGrid, 5.)
    # Only one dimensional arrays are required.
    npt.assert_raises(ValueError, RadialGrid, np.array([[5.]]))


def test_integration_general_grid():
    r"""Test normal integration on the radial grid."""
    grid = np.arange(0., 2., 0.0001)
    rad_obj = RadialGrid(grid)

    # Assume no masked values.
    model = grid
    true_answer = rad_obj.integrate(model)
    desired_answer = 2. * 2. / 2.
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-3)
