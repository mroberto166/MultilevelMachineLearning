import sobol_seq
import numpy as np
import pandas as pd


def generate_collocation_points(n_samples, n_vars, type_point, param_names=None):
    if param_names is None:
        param_names=list()
        for i in range(n_vars):
            param_names.append("Input_"+str(i))

    if type_point == "Sobol":
        collocation_points = sobol_seq.i4_sobol_generate(n_vars, n_samples)

    elif type_point == "Uniform":
        collocation_points = np.random.random((n_samples,n_vars))
    else:
        raise ValueError('Choose one option between Sobol and Uniform ')

    collocation_points_df = pd.DataFrame(collocation_points, columns = param_names)
    return collocation_points_df


def transform_data(dataframe_to_transform, mean_values, epsilon):
    transformed_data = pd.DataFrame()

    for i in range(len(dataframe_to_transform.columns)):
        transformed_data[dataframe_to_transform.columns[i]] = mean_values[i] * \
                                                              (1 + epsilon * (2 * dataframe_to_transform[dataframe_to_transform.columns[i]] - 1))

    return transformed_data
