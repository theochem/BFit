# -*- coding: utf-8 -*-
# BFit - python program that fits a convex sum of
# positive basis functions to any probability distribution. .
#
# Copyright (C) 2020 The BFit Development Team.
#
# This file is part of BFit.
#
# BFit is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# BFit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
"""Setup and Install Script."""


import os

from setuptools import setup


def get_version_info():
    """Read __version__ and DEV_CLASSIFIER from version.py, using exec, not import."""
    fn_version = os.path.join("bfit", "_version.py")
    if os.path.isfile(fn_version):
        myglobals = {}
        with open(fn_version, "r") as f:
            exec(f.read(), myglobals)  # pylint: disable=exec-used
        return myglobals["__version__"], myglobals["DEV_CLASSIFIER"]
    return "0.0.0.post0", "Development Status :: 2 - Pre-Alpha"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.md') as fhandle:
        return fhandle.read()


VERSION, DEV_CLASSIFIER = get_version_info()


setup(
    name="qc-bfit",
    version=VERSION,
    description="BFit Package",
    # description="Curve fitting algorithms for fitting basis-set functions to probabiity "
    #             "distributions.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/theochem/bfit",
    license="GNU General Public License v3.0",
    author="QC-Devs Community",
    author_email="qcdevs@gmail.com",
    package_dir={"bfit": "bfit"},
    packages=["bfit", "bfit.test"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.5", "scipy>=1.5.0", "pytest>=5.4.3", "sphinx>=2.3.0"
    ],
    package_data={
        # If any package contains *.slater files, include them:
        '': ['*.slater', '*.nwchem']
    },
    classifiers=[
         'Development Status :: 0 - Released',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
)
