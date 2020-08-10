""" Compute standard errors for parameter estimates of respy models."""

import pickle
import respy as rp
import pandas as pd
import numpy as np

# Functions needed to transform parameter vector.
from estimagic.optimization.utilities import sdcorr_params_to_matrix
from estimagic.optimization.utilities import robust_cholesky

# Functions needed to get internal criterion and parameters.
from estimagic.optimization.transform_problem import transform_problem
from estimagic.optimization.broadcast_arguments import broadcast_arguments
from estimagic.optimization.check_arguments import check_arguments
from estimagic.optimization.process_constraints import process_constraints

# Functions needed to compute covariance.
from estimagic.differentiation.numdiff_np import first_derivative
from estimagic.inference.likelihood_covs import cov_jacobian
from estimagic.inference.likelihood_covs import se_from_cov


def shocks_sdcorr_to_shocks_chol(params):
    """
    Shocks from standard deviation/correlation matrix to shocks
    from cholesky factor of variance-covariance matrix.
    """
    shocks = params.loc["shocks_sdcorr", "value"]

    # Extract choices from shock vector.
    choices = shocks.loc[shocks.index.str.contains("sd_")].index.str.replace("sd_", "")

    # Compute lower triangular cholesky factor.
    cov = sdcorr_params_to_matrix(shocks)
    cholesky = robust_cholesky(cov)
    shocks = cholesky[np.triu_indices(len(cholesky))]
    shocks = pd.DataFrame(cholesky, index=choices, columns=choices)
    shocks = shocks.where(np.tril(np.ones(shocks.shape), 0).astype(bool)).stack()

    # Reindex to fit respy parameter vector.
    indices = []
    for item in shocks.index:
        if len(set(item)) == 1:
            item = item[:-1]
        item = f"chol_{'_'.join(item)}"
        indices.append(item)
    shocks.index = indices
    shocks = pd.concat([shocks], keys=["shocks_chol"], names=["category", "name"])
    shocks.name = "value"

    return shocks.to_frame()


def params_sdcorr_to_chol(params):
    """Create new parameter vector with cholesky shocks in place of shocks from
    sd/corr matrix.
    """
    params = params.copy()
    shocks = shocks_sdcorr_to_shocks_chol(params)
    params_new = params.drop("shocks_sdcorr", level=0)
    params_new = pd.concat([params_new, shocks])
    params_index = params.index.get_level_values(0).unique()
    params_index = [
        "shocks_chol" if item == "shocks_sdcorr" else item for item in params_index
    ]
    params_new = params_new.reindex(params_index, axis=0, level=0)

    return params_new


if __name__ == "__main__":

    # Select model:
    MODEL = "kw_97_extended"

    # Increase number of draws for higher accuracy.
    SOLUTION_DRAWS = 1000
    ESTIMATION_DRAWS = 400

    if "kw_94" in MODEL:
        params, options, data = rp.get_example_model(MODEL)
        params = params[["value"]]
        constr = rp.get_parameter_constraints(MODEL)

    elif MODEL == "kw_97_basic":
        _, options, data = rp.get_example_model(MODEL)
        params = pd.read_pickle(f"{MODEL}/params_revised_basic.pkl")
        params = params[["value"]]
        constr = rp.get_parameter_constraints(MODEL)

    elif MODEL == "kw_97_extended":
        _, options, data = rp.get_example_model(MODEL)
        params = pd.read_csv(f"{MODEL}/kw_97_extended_respy_two.csv", index_col=["category", "name"])
        params = params[["value"]]

        # Workaround for a bug between respy and estimagic, will be
        # redundant eventually.
        constr = rp.get_parameter_constraints(MODEL)
        constr.remove({"query": "name == 'military_dropout'", "type": "equality"})
        constr.remove({"query": "name == 'common_co_graduate'", "type": "equality"})
        constr.remove({"query": "name == 'common_hs_graduate'", "type": "equality"})
        constr += [
            {
                "loc": [
                    ("nonpec_white_collar", "military_dropout"),
                    ("nonpec_blue_collar", "military_dropout"),
                    ("nonpec_school", "military_dropout"),
                    ("nonpec_home", "military_dropout"),
                ],
                "type": "equality",
            },
            {
                "loc": [
                    ("nonpec_white_collar", "common_co_graduate"),
                    ("nonpec_blue_collar", "common_co_graduate"),
                    ("nonpec_military", "common_co_graduate"),
                    ("nonpec_school", "common_co_graduate"),
                    ("nonpec_home", "common_co_graduate"),
                ],
                "type": "equality",
            },
            {
                "loc": [
                    ("nonpec_white_collar", "common_hs_graduate"),
                    ("nonpec_blue_collar", "common_hs_graduate"),
                    ("nonpec_military", "common_hs_graduate"),
                    ("nonpec_school", "common_hs_graduate"),
                    ("nonpec_home", "common_hs_graduate"),
                ],
                "type": "equality",
            },
        ]

    # This constraint is not needed and conflicts with the cholesky
    # shock parameter vector created in the next line.
    constr.remove({"loc": "shocks_sdcorr", "type": "sdcorr"})

    # Get parameter vector with cholesky shocks.
    params_new = params_sdcorr_to_chol(params)

    # Get likelihood functon.
    options["solution_draws"] = SOLUTION_DRAWS
    options["estimation_draws"] = ESTIMATION_DRAWS
    log_likelihood_contrib = rp.get_crit_func(
        params_new, options, data, return_scalar=False
    )

    # Get internal criterion and params.
    arguments = broadcast_arguments(
        criterion=log_likelihood_contrib,
        params=params_new,
        algorithm="nlopt_bobyqa",
        criterion_kwargs=None,
        constraints=constr,
        general_options=None,
        algo_options=None,
        gradient=None,
        gradient_kwargs=None,
        gradient_options=None,
        logging=False,
        log_options=None,
        dashboard=False,
        dash_options=None,
    )
    check_arguments(arguments)
    for single_arg in arguments:
        optim_kwargs, database_path, result_kwargs = transform_problem(**single_arg)

    # Compute jacobian.
    criterion_internal = optim_kwargs["internal_criterion"]
    params_internal = optim_kwargs["internal_params"]
    jacobian = first_derivative(criterion_internal, params_internal, n_cores=1)

    with open(f"{MODEL}/jacobian_{MODEL}.pkl", "wb") as out:
        pickle.dump(jacobian, out)

    # Compute covariance.
    cov_jac = cov_jacobian(jacobian)
    with open(f"{MODEL}/covariance_{MODEL}_numpy.pkl", "wb") as outfile:
        pickle.dump(cov_jac, outfile)

    # Get indexed internal parameter vector to extract index for covariance matrix.
    params_new["lower"] = -(np.inf)
    params_new["upper"] = np.inf
    _, params_internal_indexed = process_constraints(constr, params_new)
    indices_params_internal = params_internal_indexed[
        params_internal_indexed._internal_free
    ].index
    cov = pd.DataFrame(
        cov_jac, index=indices_params_internal, columns=indices_params_internal
    )
    with open(f"{MODEL}/covariance_{MODEL}.pkl", "wb") as outfile:
        pickle.dump(cov, outfile)

    # Compute standard errors
    standard_errors = se_from_cov(cov_jac)
    se = pd.DataFrame(standard_errors, index=indices_params_internal)
    params_new["se"] = se
    params_new[["value", "se"]].to_pickle(f"{MODEL}/params_{MODEL}_se.pkl")
