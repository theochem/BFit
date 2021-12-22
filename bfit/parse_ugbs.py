
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
