import ODE as Solver
import GenerateDataClass as Generator
import config
from joblib import Parallel, delayed
import multiprocessing
import time
import matplotlib.pyplot as plt


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
point_type = "Sobol"

parameters_name = ["density", "radius", "drag_coefficient", "mass", "h", "alpha", "v0"]

n_samples = 8000
fig = plt.figure()
with open('./CaseStudies/Parabolic/Data/ComputationalTime.txt', 'w') as file:
    file.write("timestep,comp_time\n")
    time_sample = list()
    for delta_t in delta_t_vec:
        now = time.time()
        collocation_points = Generator.generate_collocation_points(n_samples, len(parameters_name), point_type, parameters_name )
        collocation_points_transformed = Generator.transform_data(collocation_points, means_values, epsilon)

        vector_output = Parallel(n_jobs=num_cores)(
            delayed(Solver.solve_for_input_data)(collocation_points_transformed, i, delta_t) for i in range(len(collocation_points_transformed)))
        collocation_points["x_max"] = vector_output
        end = time.time() - now
        print("Samples: ", n_samples)
        print("Time-step: ", delta_t)
        print("Time: ", end)
        file.write(str(delta_t)+","+str(end/n_samples)+"\n")
        time_sample.append(end)
