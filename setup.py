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
