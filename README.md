# FittingBasisSets

FittingBasisSets is a python program that used to fit a convex sum of basis
sets to any probability distribution. Primarily used for ab-intio quantum
chemistry calculations with basis sets being gaussian.


# What it contains
Two optimization procedures were used to solve this problem:
Least-Squares
    
Kullback-Leibler Divergence


Greedy Fitting Algorithm


For more info, please see the paper cited below or go to the respective
folders.

# Dependences


# Installation


# Examples


# Why is this needed?
The basis sets of importance are, slater densities and gaussian densities.
Slater densities are more accurate, due to the cusp at r=0, however are 
numerically harder to integrate. On the other hand,
gaussian basis sets are not accurate, due to the lack of cusp at r=0, however 
are easier to integrate.
Curve-fitting procedures is used to convert between the two however modeling
this problem as a standard least-squares does not suffice to solve it.

# More Info

# Citing

