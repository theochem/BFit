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
from setuptools import setup, find_packages

setup(
    name="fitting",
    version="0.1",
    description="Curve fitting algorithms for fitting basis-set functions to probabiity "
                "distributions.",
    author="Ali Tehrani, Farnaz Heidar-Zadeh and Paul Ayers",
    author_email="alirezatehrani24@gmail.com and ayers@mcmaster.ca",
    install_requires=[
        "numpy", "scipy", "matplotlib", "nose"
    ],
    packages=find_packages('fitting'),
    package_data={
        # If any package contains *.slater files, include them:
        '': ['*.slater', '*.nwchem']
    }
)
