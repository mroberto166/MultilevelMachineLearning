import os
import UtilsNetwork as Utils
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
import ast
import itertools
import sys
from termcolor import colored
import pandas as pd
import joblib
import numpy as np

# df_final_model = pd.DataFrame(columns=["level_c", "level_f", "samples", "scaler", "method"])


def get_finest_score(level, sample, norm_string, scaler):
    folder_name = Utils.set_model_folder_name(keyword, variable_name, 0, 0, level, sample)

    folder_path = "CaseStudies/" + case_study + "/Models/" + folder_name
    if not os.path.exists(folder_path + "_GP"):
        print(colored("\nEvaluating Gaussian model model at level " + str(level), 'yellow', attrs=['bold']))
        Utils.call_GaussianProcess(keyword, variable_name, sample, level, 0, 0, norm_string, scaler, point)
    else:
        print(colored("\nThe Gaussian model at level " + str(level) + " already exists", 'yellow', attrs=['bold']))

    if not os.path.exists(folder_path):
        print(colored("\nEvaluating Network model model at level " + str(level), 'yellow', attrs=['bold']))

        parameter_grid_fin = Utils.get_network_conf(keyword, variable_name, level, 0, 0)
        setting = list(itertools.product(*parameter_grid_fin.values()))

        Utils.call_NeuralNetwork_cluster(keyword, sample, loss, folder_path,
                                             variable_name, 0, 0, level,
                                             norm_string, validation_size, selection_method, scaler, setting[0], point)

    else:
        print(colored("\nThe Network model at level " + str(level) + " already exists", 'yellow', attrs=['bold']))


os.system('color')
if sys.platform == "win32":
    python = os.environ['PYTHON36']


validation_size = 0.1
selection_method = "validation_loss"

# ==========================
# Model Info
levels = ast.literal_eval(sys.argv[1])
samples_vec = ast.literal_eval(sys.argv[2])
keyword = sys.argv[3]
variable_name = sys.argv[4]
loss = sys.argv[5]
norm_vec = ast.literal_eval(sys.argv[6])
norm_string_finest = sys.argv[7]
scaler_finest = sys.argv[8]
scaler_vec = ast.literal_eval(sys.argv[9])
point = sys.argv[10]
model = sys.argv[11]

# ==========================
keyword_diff = keyword + "_diff"
sample_0 = samples_vec[-1]
level_0 = levels[-1]
norm_string_0 = norm_vec[-1]
scaler_0 = scaler_vec[-1]
# type_model_0 = types_model[-1]

sample_finest = samples_vec[0]
level_finest = levels[0]


n_input = None

print(colored("\n====================", 'yellow', attrs=['bold']))
print(colored("Initial Info", 'yellow', attrs=['bold']))
print("Samples for the function:", sample_0)
print("Keyword:", keyword)
print("Variable Name:", variable_name)
print("Loss:", loss)
print(colored("====================", 'yellow', attrs=['bold']))


if "parab" in keyword:
    n_input = 7
    case_study = "Parabolic"
elif "shock" in keyword:
    n_input = 6
    case_study = "ShockTube"
elif "airf" in keyword:
    n_input = 6
    case_study = "Airfoil"
else:
    raise ValueError("Chose one option between parab, airf and shock")


# get_finest_score(level_finest, sample_finest, norm_string_finest, scaler_finest)


# Build molder for the function at the coarsest grid
time = 0
func_folder_name = Utils.set_model_folder_name(keyword, variable_name, 0, 0, level_0, sample_0)
func_folder_name = func_folder_name #+ prefix
func_folder_path = "CaseStudies/" + case_study + "/Models/" + func_folder_name

print(colored("\n\n==========================================================", 'grey', attrs=['bold']))
if not os.path.exists(func_folder_path + "_GP"):
    print(colored("\nGaussian model at level " + str(level_0) + " does not exists: Training", 'yellow', attrs=['bold']))

    Utils.call_GaussianProcess(keyword, variable_name, sample_0, level_0, 0, 0, norm_string_0, scaler_0, point)
else:
    print(colored("\nGaussian model at level " + str(level_0) + " already exists", 'yellow', attrs=['bold']))


if not os.path.exists(func_folder_path):
    print(colored("\nNetwork model at level " + str(level_0) + " does not exists: Training", 'yellow', attrs=['bold']))

    parameter_grid = Utils.get_network_conf(keyword, variable_name, level_0, 0, 0)

    settings = list(itertools.product(*parameter_grid.values()))

    Utils.call_NeuralNetwork_cluster(keyword, sample_0, loss, func_folder_path,
                                         variable_name, 0, 0, level_0,
                                         norm_string_0, validation_size, selection_method, scaler_0, settings[0], point)
else:
    print(colored("\nNetwork model at level " + str(level_0) + " already exists", 'yellow', attrs=['bold']))
time_0 = Utils.compute_time(keyword, 0, 0, level_0, sample_0)
if sample_0 > 500 and model == "ENS":
    time = time + time_0*1.1
else:
    time = time + time_0


# Build the models for the difference functions
for i in range(len(levels)-1):
    print(colored("\n\n==========================================================", 'grey', attrs=['bold']))
    samples = samples_vec[i]
    level_c = levels[i+1]
    level_f = levels[i]
    norm_string_diff = norm_vec[i]
    scaler_diff = scaler_vec[i]

    diff_folder_name = Utils.set_model_folder_name(keyword_diff, variable_name, level_c, level_f, level_f, samples)
    diff_folder_name = diff_folder_name
    diff_folder_path = "CaseStudies/" + case_study + "/Models/" + diff_folder_name

    if not os.path.exists(diff_folder_path + "_GP"):
        print(colored("The difference Gaussian model at levels " + str(level_c) + " and " + str(level_f) + " does not exists: Training", 'yellow', attrs=['bold']))
        Utils.call_GaussianProcess(keyword_diff, variable_name, samples, level_f, level_c, level_f, norm_string_diff, scaler_diff, point)

    else:
        print(colored("The difference Gaussian model at levels " + str(level_c) + " and " + str(level_f) + " already exists", 'yellow', attrs=['bold']))

    if not os.path.exists(diff_folder_path):
        print(colored("The difference Network model at levels " + str(level_c) + " and " + str(level_f) + " does not exists: Training", 'yellow', attrs=['bold']))

        # Run search network properly to get these values
        parameter_grid = Utils.get_network_conf(keyword_diff, variable_name, 0, level_c, level_f)
        settings = list(itertools.product(*parameter_grid.values()))
        Utils.call_NeuralNetwork_cluster(keyword_diff, samples, loss, diff_folder_path,
                                         variable_name, level_c, level_f, level_f,
                                         norm_string_diff, validation_size, selection_method, scaler_diff, settings[0], point)
    else:
        print(colored("The difference Network model at levels " + str(level_c) + " and " + str(level_f) + " already exists", 'yellow', attrs=['bold']))
    time_diff = Utils.compute_time(keyword_diff, level_f, level_c, 0, samples)
    if samples > 500 and model == "ENS":
        time = time + time_diff*1.1
    else:
        time = time + time_diff


time_finest_1_samp = Utils.compute_time(keyword,0,0,level_finest,1)
n_samp_fin = int(round(time/time_finest_1_samp,0))
print(n_samp_fin)
get_finest_score(level_finest, n_samp_fin, norm_string_finest, scaler_finest)