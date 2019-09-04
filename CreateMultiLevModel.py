import os
import UtilsNetwork as Utils
from distutils.dir_util import copy_tree
import ast
import itertools
import sys
from termcolor import colored
import pandas as pd
import joblib
import numpy as np

# df_final_model = pd.DataFrame(columns=["level_c", "level_f", "samples", "scaler", "method"])


def get_finest_score(level, sample, norm_string, scaler, mod):
    folder_name = Utils.set_model_folder_name(keyword, variable_name, 0, 0, level, sample)

    folder_path = "CaseStudies/" + case_study + "/Models/" + folder_name
    print("Looking for:" + folder_path + "_GP")
    if not os.path.exists(folder_path + "_GP"):
        print(colored("\nEvaluating Gaussian model model at level " + str(level), 'yellow', attrs=['bold']))
        Utils.call_GaussianProcess(keyword, variable_name, sample, level, 0, 0, norm_string, scaler, point)
    else:
        print(colored("\nThe Gaussian model at level " + str(level) + " already exists", 'yellow', attrs=['bold']))

    score_gauss = pd.read_csv(folder_path + "_GP/Score.txt", sep=",", header=0)
    MPE_gauss = score_gauss.MPE.values[0]
    print("Looking for:" + folder_path)
    if not os.path.exists(folder_path):
        print(colored("\nEvaluating Network model model at level " + str(level), 'yellow', attrs=['bold']))

        parameter_grid_fin = Utils.get_network_conf(keyword, variable_name, level, 0, 0)
        setting = list(itertools.product(*parameter_grid_fin.values()))

        Utils.call_NeuralNetwork_cluster(keyword, sample, loss, folder_path,
                                         variable_name, 0, 0, level,
                                         norm_string, validation_size, selection_method, scaler, setting[0], point)

    else:
        print(colored("\nThe Network model at level " + str(level) + " already exists", 'yellow', attrs=['bold']))

    size_validation_ens = 0.1
    X, y, _, _, _, _ = Utils.get_data(keyword, "all", variable_name, level, n_input=n_input, model_path_folder=None, normalize=False, point=point)
    sample_validation_ens = int(samples * size_validation_ens)
    if sample_validation_ens == 0:
        sample_validation_ens = 1
    print(sample_validation_ens)

    X_ens = X[sample:sample + sample_validation_ens, :]
    y_ens = y[sample:sample + sample_validation_ens]
    print(sample)

    X = X[sample:, :]
    y = y[sample:]

    y = y.reshape(-1, )
    y_ens = y_ens.reshape(-1, )

    model_GP = joblib.load(folder_path + "_GP" + "/model_GP.sav")
    model_net = Utils.load_data(folder_path)

    y_net = model_net.predict(X)
    y_GP = model_GP.predict(X)

    y_net_ens = model_net.predict(X_ens)
    y_GP_ens = model_GP.predict(X_ens)

    minmax = pd.read_csv(folder_path + "/MinMax.txt", header=0)
    min_val = minmax.Min.values[0]
    max_val = minmax.Max.values[0]

    if scaler == "m":
        y_net = y_net * (max_val - min_val) + min_val
        y_GP = y_GP * (max_val - min_val) + min_val

        y_net_ens = y_net_ens * (max_val - min_val) + min_val
        y_GP_ens = y_GP_ens * (max_val - min_val) + min_val
    elif scaler == "s":
        y_net = y_net * max_val + min_val
        y_GP = y_GP * max_val + min_val

        y_net_ens = y_net_ens * max_val + min_val
        y_GP_ens = y_GP_ens * max_val + min_val

    y_net = y_net.reshape(-1, )
    y_GP = y_GP.reshape(-1, )

    print("Network:", Utils.compute_mean_prediction_error(y, y_net, 2) * 100)
    print("Gaussian:", Utils.compute_mean_prediction_error(y, y_GP, 2) * 100)

    y_net_ens = y_net_ens.reshape(-1, )
    y_GP_ens = y_GP_ens.reshape(-1, )

    time = Utils.compute_time(keyword, 0, 0, level, sample)

    if mod == "ENS":

        y_pred = (y_GP + y_net) / 2
        model_type = "Mean"

        if sample > 500:
            print("Doing Ensemble")
            y_pred_mods_ens = [y_net_ens, y_GP_ens]
            y_pred_mods_ens = np.array(y_pred_mods_ens).transpose()

            y_pred_mods = [y_net, y_GP]
            y_pred_mods = np.array(y_pred_mods).transpose()

            ensemb = Utils.ensemble_model(y_pred_mods_ens, y_ens)

            y_ensemble = ensemb.predict(y_pred_mods)
            y_pred = y_ensemble
            model_type = "Ensemble"
            time = time + time * size_validation_ens

    elif mod == "GP":
        y_pred = y_GP
        model_type = "GP"
    elif mod == "NET":
        y_pred = y_net
        model_type = "NET"
    else:
        raise ValueError("Check mod argument: NET, GP, ENS")

    MPE = Utils.compute_mean_prediction_error(y, y_pred, 2) * 100
    SPE = Utils.compute_prediction_error_variance(y, y_pred, 2) * 100
    print(MPE)
    print(SPE)

    with open(assembled_model_folder_path + '/Score_fin.txt', 'w') as file_:
        file_.write("MPE,SPE\n")
        file_.write(str(MPE))
        file_.write(",")
        file_.write(str(SPE))

    with open(assembled_model_folder_path + '/Model_Finest.txt', 'w') as file_:
        file_.write(str(model_type))

    with open(assembled_model_folder_path + '/Time_Finest.txt', 'w') as file_:
        file_.write(str(time))


os.system('color')
if sys.platform == "win32":
    python = os.environ['PYTHON36']
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
depth = round(Utils.compute_mean_depth(levels), 1)
# ==========================
keyword_diff = keyword + "_diff"
validation_size = 0.15
selection_method = "validation_loss"
sample_0 = samples_vec[-1]
level_0 = levels[-1]
norm_string_0 = norm_vec[-1]
scaler_0 = scaler_vec[-1]
# type_model_0 = types_model[-1]

sample_finest = samples_vec[0]
level_finest = levels[0]


n_input = None
assembled_model_folder = "Depth_"+str(len(levels))+"_"+variable_name+"_"+str(depth)+"_"+str(sample_0)+"_"+str(sample_finest)


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


# Create assembled model folder
assembled_model_folder_path = "CaseStudies/" + case_study + "/Models/" + assembled_model_folder
print(colored("\nCreating folder: " + assembled_model_folder_path, 'green', attrs=['bold']))
os.mkdir(assembled_model_folder_path)

# Build molder for the function at the coarsest grid
time = 0
func_folder_name = Utils.set_model_folder_name(keyword, variable_name, 0, 0, level_0, sample_0)
func_folder_name = func_folder_name #+ prefix
func_folder_path = "CaseStudies/" + case_study + "/Models/" + func_folder_name

print(colored("\n\n==========================================================", 'grey', attrs=['bold']))


print("Looking for:" + func_folder_path + "_GP")
if not os.path.exists(func_folder_path + "_GP"):

    print(colored("\nGaussian model at level " + str(level_0) + " does not exists: Training", 'yellow', attrs=['bold']))

    Utils.call_GaussianProcess(keyword, variable_name, sample_0, level_0, 0, 0, norm_string_0, scaler_0, point)
else:
    print(colored("\nGaussian model at level " + str(level_0) + " already exists", 'yellow', attrs=['bold']))


print("Looking for:" + func_folder_path)
if not os.path.exists(func_folder_path):
    print(colored("\nNetwork model at level " + str(level_0) + " does not exists: Training", 'yellow', attrs=['bold']))

    parameter_grid = Utils.get_network_conf(keyword, variable_name, level_0, 0, 0)
    settings = list(itertools.product(*parameter_grid.values()))
    Utils.call_NeuralNetwork_cluster(keyword, sample_0, loss, func_folder_path,
                                         variable_name, 0, 0, level_0,
                                         norm_string_0, validation_size, selection_method, scaler_0, settings[0], point)
else:
    print(colored("\nNetwork model at level " + str(level_0) + " already exists", 'yellow', attrs=['bold']))

folder_in_assembled = assembled_model_folder_path + "/" + func_folder_name
os.mkdir(folder_in_assembled)
print(colored("Copying : " + func_folder_path + " into " + folder_in_assembled, 'green', attrs=['bold']))
copy_tree(func_folder_path, folder_in_assembled)
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

    print("Looking for:" + diff_folder_path + "_GP")
    if not os.path.exists(diff_folder_path + "_GP"):
        print(colored("The difference Gaussian model at levels " + str(level_c) + " and " + str(level_f) + " does not exists: Training", 'yellow', attrs=['bold']))
        Utils.call_GaussianProcess(keyword_diff, variable_name, samples, level_f, level_c, level_f, norm_string_diff, scaler_diff, point)

    else:
        print(colored("The difference Gaussian model at levels " + str(level_c) + " and " + str(level_f) + " already exists", 'yellow', attrs=['bold']))

    print("Looking for:" + diff_folder_path)
    if not os.path.exists(diff_folder_path):
        print(colored("The difference Network model at levels " + str(level_c) + " and " + str(level_f) + " does not exists: Training", 'yellow', attrs=['bold']))
        parameter_grid = Utils.get_network_conf(keyword_diff, variable_name, 0, level_c, level_f)
        settings = list(itertools.product(*parameter_grid.values()))
        Utils.call_NeuralNetwork_cluster(keyword_diff, samples, loss, diff_folder_path,
                                         variable_name, level_c, level_f, level_f,
                                         norm_string_diff, validation_size, selection_method, scaler_diff, settings[0], point)

    else:
        print(colored("The difference Network model at levels " + str(level_c) + " and " + str(level_f) + " already exists", 'yellow', attrs=['bold']))
    # Move the folder model into the assembled folder
    folder_in_assembled = assembled_model_folder_path + "/" + diff_folder_name
    os.mkdir(folder_in_assembled)
    print(colored("Copying : " + diff_folder_path + " into " + folder_in_assembled, 'green', attrs=['bold']))
    copy_tree(diff_folder_path, folder_in_assembled)
    time_diff = Utils.compute_time(keyword_diff, level_f, level_c, 0, samples)
    if samples > 500 and model == "ENS":
        time = time + time_diff*1.1
    else:
        time = time + time_diff


print("\nTotal time for the model:")
print(time)
time_finest_1_samp = Utils.compute_time(keyword, 0, 0, level_finest, 1)
n_samp_fin = int(round(time/time_finest_1_samp, 0))
print(n_samp_fin)
get_finest_score(level_finest, n_samp_fin, norm_string_finest, scaler_finest, model)
with open(assembled_model_folder_path + '/Time.txt', 'w') as file:
    file.write(str(time))

