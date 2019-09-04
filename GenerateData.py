import ODE as solver
import GenerateDataClass as Generator
import config
from joblib import Parallel, delayed
import multiprocessing
import time
import numpy as np

np.random.seed(42)


num_cores = multiprocessing.cpu_count()

means_values = list()
means_values.append(config.density_mean)
means_values.append(config.radius_mean)
means_values.append(config.drag_mean)
means_values.append(config.mass_mean)
means_values.append(config.h_mean)
means_values.append(config.alpha_mean)
means_values.append(config.v0_mean)

epsilon = config.epsilon
delta_t_vec = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125]

parameters_name = ["density", "radius", "drag_coefficient", "mass", "h", "alpha", "v0"]

n_samples = 4000
collocation_points = Generator.generate_collocation_points(n_samples, len(parameters_name), "Uniform", parameters_name )
collocation_points_transformed = Generator.transform_data(collocation_points, means_values, epsilon)
for j in range(len(delta_t_vec)):
    now = time.time()
    vector_output = Parallel(n_jobs=num_cores)(
        delayed(solver.solve_for_input_data)(collocation_points_transformed, i, delta_t_vec[j]) for i in range(len(collocation_points_transformed)))
    collocation_points["x_max"] = vector_output
    # collocation_points.to_csv("./Files/solution_sobol_deltaT_bad.csv", header=True, index=False, sep=",")
    collocation_points.to_csv("./CaseStudies/Parabolic/Data/solution_random_deltaT_"+str(j)+".csv", header=True, index=False, sep=",")
    end = time.time() - now
    print(end)
