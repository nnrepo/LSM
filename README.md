

INSTALLATION:
-------
Set in cmake the NeMo-0.7.2, boost_1_55_0 and Eigen3 directories.

Then run make.
To train the LSM run:
./main

This will train an LSM and stack the results into a best.m matlab file (for fast loading the results into matlab).



Implementation details for the liquid are given in:
MAASS, W. "Real-time Computing Without Stable States: A New Framework for Neural Computation Based  on perturbations." Neural Computations 14 (2002): 2531-2560.
