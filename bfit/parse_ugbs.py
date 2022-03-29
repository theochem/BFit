# -*- coding: utf-8 -*-
# BFit is a Python library for fitting a convex sum of Gaussian
# functions to any probability distribution
#
# Copyright (C) 2020- The QC-Devs Community
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
r"""Obtain UGBS exponents from ygbs file."""
import inspect
import os

__all__ = ["get_ugbs_exponents"]


def get_ugbs_exponents(element):
    r"""
    Get the UGBS exponents of a element.

    Parameters
    ----------
    element : str
        The element whose UGBS exponents are required.

    Returns
    -------
    dict
        Dictionary with keys "S" or "P" type orbitals and the items are the exponents for that
        shell.

    """
    assert isinstance(element, str)
    path_to_function = os.path.abspath(inspect.getfile(get_ugbs_exponents))
    path_to_function = path_to_function[:-13]  # Remove parse_ugbs.py <- 13 characters
    file_path = path_to_function + r"data/ygbs"
    output = {"S" : [], "P" : [], "D" : [], "F" : []}
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            split_words = line.strip().split()
            if len(split_words) > 0 and split_words[0].lower() == element.lower():
                next_line = f.readline().strip().split()
                output[split_words[1]].append(float(next_line[0]))
    return output
