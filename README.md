# respy-se-collection

This repository contains standard errors and covariance matrices I have computed for models in the respy interface. All files can be created by selecting the desired respy model in `run_standard_errors.py` and running the script. Derivates are computed on estimagic's inner versions of the criterion function and parameter dataframe by taking constraints that are also used for optimization into consideration.

The following files will be created:

- `jacobian_<model>.pkl` : Jacbian derivative (pandas.DataFrame) based on inner likelihood and inner parameter dataframe (cholesky factor version).
- `covariance_<model>.pkl`: (Inner) covaraince matrix as an indexed pandas.DataFrame.
- `covariance_<model>_numpy.pkl`: (Inner) covaraince matrix as a numpy.array. 
- `params_<model>_se.pkl`: Respy parameter dataframe (cholesky factor version) with column `se` containing computed standard errors.

All files currently saved to the repository are created using 1000 solution and 400 estimation draws, which is double the default settings. The parameters used for the **KW97** models are new estimates that have not been integrated into the respy iterface yet and are thus saved in their respective folers as pickle files.

Feedback is very welcome. For questions and comments please raise an issue or contact me directly.

Further information
---------------------
respy documentation: https://respy.readthedocs.io

estimagic documentation: https://estimagic.readthedocs.io


