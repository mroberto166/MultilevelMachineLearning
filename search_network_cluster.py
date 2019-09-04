#!/usr/bin/env python
import pandas as pd
import itertools
import UtilsNetwork as Utils
import os
import pprint
import sys
from matplotlib import rc
from termcolor import colored
os.system('color')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
print(sys.argv)
# =========================================
# Data Initialization
seed = 42
# Net.seed_random_number(seed)
keyword = sys.argv[1]
variable_name = sys.argv[2]
samples = int(sys.argv[3])
loss = sys.argv[4]
level_single = sys.argv[5]
level_c = sys.argv[6]
level_f = sys.argv[7]
selection_method = sys.argv[8]
validation_size = float(sys.argv[9])
string_norm = sys.argv[10]
scaler = sys.argv[11]
search_folder = sys.argv[12]
point=sys.argv[13]

n_input = 0

# =========================================
# Default values to search reg param: 0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
# Default values to search learning rate: 0.001, 0.01

# ====================================================
if string_norm == "true":
    norm = True
elif string_norm == "false":
    norm = False
else:
    raise ValueError("Norm can be 'true' or 'false'")


# ====================================================
if keyword == "parab":
    n_input = 7
    case_folder = "Parabolic"
    parameter_grid = {
        "regularization_parameter": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        "kernel_regularizer": ["L1", "L2"],
        "learning_rate": [0.01],
        "hidden_layers": [5],
        "neurons": [12],
        "dropout_value": [0]
    }
elif keyword == "parab_diff":
    n_input = 7
    case_folder = "Parabolic"
    parameter_grid = {
        "regularization_parameter": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        "kernel_regularizer": ["L1", "L2"],
        "learning_rate": [0.01],
        "hidden_layers": [5],
        "neurons": [10],
        "dropout_value": [0]
    }

elif keyword == "shock":
    n_input = 6
    case_folder = "ShockTube"
    parameter_grid = {
        "regularization_parameter": [1e-5],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [4],
        "neurons": [10],
        "dropout_value": [0]
    }
elif keyword == "shock_diff":
    n_input = 6
    case_folder = "ShockTube"
    parameter_grid = {
        "regularization_parameter": [1e-5],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [7],
        "neurons": [10],
        "dropout_value": [0]
    }
elif keyword == "airf_diff":
    n_input = 6
    case_folder = "Airfoil"
    parameter_grid = {
        "regularization_parameter": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        "kernel_regularizer": ["L1","L2"],
        "learning_rate": [0.001, 0.005, 0.01],
        "hidden_layers": [8, 12, 16, 20, 24],
        "neurons": [8, 12, 16, 20, 24],
        "dropout_value": [0]
    }

elif keyword == "airf":
    n_input = 6
    case_folder = "Airfoil"
    parameter_grid = {
        "regularization_parameter": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.001, 0.005, 0.01],
        "hidden_layers": [8, 12, 16, 20, 24],
        "neurons": [8, 12, 16, 20, 24],
        "dropout_value": [0]
    }
    # 4.6e-08#1e-7, 5e-7, 1e-6, 5e-6, 1e-5,
else:
    raise ValueError("Chose one option between parab and shock")

# ====================================================


# ====================================================
if "diff" in keyword:
    folder_name = variable_name + "_" + str(level_c) + str(level_f) + "_" + str(samples) + "_" + keyword
else:
    folder_name = variable_name + "_" + str(level_single) + "_" + str(samples) + "_" + keyword

model_path_folder = "./CaseStudies/" + case_folder + "/Models/" + search_folder
os.mkdir(model_path_folder)

# ====================================================
if "diff" in keyword:
    _, _, X_test, y_test, min_val, max_val = Utils.get_data_diff(keyword, samples, variable_name, level_c, level_f, n_input=n_input,
                                                                 model_path_folder=model_path_folder, normalize=norm, scaler=scaler, point=point)
else:
    _, _, X_test, y_test, min_val, max_val = Utils.get_data(keyword, samples, variable_name, level_single, n_input,
                                                            model_path_folder=model_path_folder, normalize=norm, scaler=scaler, point=point)

# ====================================================
print("\n")
print("\n")
print(colored("*******************************************************", 'yellow', attrs=['bold']))
print(colored("Info for the grid search:", 'yellow', attrs=['bold']))
print("Keyword:", keyword)
print("variable_name:", variable_name)
print("samples:", samples)
print("loss:", loss)
print("level_single:", level_single)
print("level_c:", level_c)
print("level_f:", level_f)
print("selection_method:", selection_method)
print("validation_size:", validation_size)
print("norm:", norm)
print(colored("*******************************************************", 'yellow', attrs=['bold']))
# ====================================================

settings = list(itertools.product(*parameter_grid.values()))
previous_error = 10
best_network = None
best_setting = None
best_network_info = None


i = 0
for setup in settings:
    print("\n")
    print(colored("=================================================", 'grey', attrs=['bold']))
    print(colored("Current Setting: " + str(setup), 'grey', attrs=['bold']))
    model_path_folder_set = model_path_folder + os.sep + folder_name + "_" + str(i)

    Utils.call_NeuralNetwork_cluster(keyword, samples, loss, model_path_folder_set,
                                     variable_name, level_c, level_f, level_single,
                                     string_norm, validation_size, selection_method, scaler, setup, point)

    i = i+1

print(colored("\n*******************************************************", 'yellow', attrs=['bold']))



