import ODE as solver
from joblib import Parallel, delayed
import multiprocessing
import time
import GenerateDataClass as generator
import config


# Computational time
starting_time = time.time()
# Get number of cores
num_cores = multiprocessing.cpu_count()
# Get mean values for input transformation
means_values = list()
means_values.append(config.density_mean)
means_values.append(config.radius_mean)
means_values.append(config.drag_mean)
means_values.append(config.mass_mean)
means_values.append(config.h_mean)
means_values.append(config.alpha_mean)
means_values.append(config.v0_mean)
epsilon = config.epsilon

n_samples = 20000
parameters_name = config.parameters_name
n_vars = len(parameters_name)
type_point = "Uniform"
delta_t = 0.001

collocation_points = generator.generate_collocation_points(n_samples, n_vars, type_point, param_names=parameters_name)
collocation_points_transformed = generator.transform_data(collocation_points, means_values, epsilon)

vector_output = Parallel(n_jobs=num_cores)(
    delayed(solver.solve_for_input_data)(collocation_points_transformed, i, delta_t) for i in range(len(collocation_points_transformed)))

collocation_points["x_max"] = vector_output
collocation_points.to_csv("./Files/ReferenceUQ/ref_solution_20k.csv", header=True, index=False, sep=",")
print("Computational time: ", time.time() - starting_time)



