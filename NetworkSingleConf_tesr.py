import matplotlib.pyplot as plt
import pandas as pd
import NetworkClass as Net
import itertools
import UtilsNetwork as Utils
import sys
import numpy as np
import os
from matplotlib import rc
import joblib
from termcolor import colored
from sklearn.model_selection import KFold
os.system('color')

#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#rc('text', usetex=True)

seed = 42

# Net.seed_random_number(seed)
print(sys.argv)
keyword = sys.argv[1]
samples = int(sys.argv[2])
loss = sys.argv[3]
regularization_parameter = float(sys.argv[4])#/(samples*(1-validation_size))
kernel_regularizer = sys.argv[5]
learning_rate = float(sys.argv[6])
hidden_layers = int(sys.argv[7])
neurons_hidden_layer = int(sys.argv[8])
dropout_value = float(sys.argv[9])
# previous_error = float(sys.argv[10])
folder_path = sys.argv[10]
variable_name = sys.argv[11]
level_c = (sys.argv[12])
level_f = (sys.argv[13])
level_single = (sys.argv[14])
string_norm = sys.argv[15]
validation_size = float(sys.argv[16])
selection = sys.argv[17]
scaler = sys.argv[18]
point = sys.argv[19]
# validation_size = 0.15
# selection = "validation_loss"

scaler = scaler.replace("'","")
string_norm = string_norm.replace("'","")

if validation_size>=1 or validation_size == 0:
    validation_size = int(1)

# ====================================================
# Model folder path is the path of the path of the folder for the current setup

if string_norm == "true":
    norm = True
elif string_norm == "false":
    norm = False
else:
    raise ValueError("Norm can be 'true' or 'false'")

Net.seed_random_number(seed)
Net.single_thread()
# setup = [regularization_parameter, kernel_regularizer, learning_rate, hidden_layers, neurons_hidden_layer, dropout_value]
setup = {
        "regularization_parameter": [regularization_parameter],
        "kernel_regularizer": [kernel_regularizer],
        "learning_rate": [learning_rate],
        "hidden_layers": [hidden_layers],
        "neurons": [neurons_hidden_layer],
        "dropout_value": [dropout_value]
    }

if scaler == "m":
    output_activation = "relu"
elif scaler == "s":
    output_activation = None
else:
    raise ValueError("Select one scaler between MinMax (m) and Standard (s)")


# ====================================================
if keyword == "parab":
    n_input = 7
    case_folder = "Parabolic"
elif keyword == "parab_diff":
    n_input = 7
    case_folder = "Parabolic"
elif keyword == "shock":
    n_input = 6
    case_folder = "ShockTube"
elif keyword == "shock_diff":
    n_input = 6
    case_folder = "ShockTube"
elif keyword == "airf_diff":
    n_input = 6
    case_folder = "Airfoil"
elif keyword == "airf":
    n_input = 6
    case_folder = "Airfoil"
else:
    raise ValueError("Chose one option between parab and shock")

# ====================================================

os.mkdir(folder_path)

if "diff" in keyword:
    X, y, X_test, y_test, min_val, max_val = Utils.get_data_diff(keyword, samples, variable_name, level_c, level_f, n_input,
                                                                 model_path_folder=folder_path, normalize=norm, scaler=scaler, point=point)
else:
    X, y, X_test, y_test, min_val, max_val = Utils.get_data(keyword, samples, variable_name, level_single, n_input,
                                                            model_path_folder=folder_path, normalize=norm, scaler=scaler, point=point)

mean_error_reg, std_error_reg, model_reg = Utils.linear_regression(keyword, variable_name, X.shape[0], level_c, level_f, level_single, n_input, norm, scaler, point)
filename = folder_path + '/model_reg.sav'
joblib.dump(model_reg, filename)
batch_size = 256 if X.shape[0] > 256 else X.shape[0]
network = Net.SetNetworkInfo(epochs=10000,
                             batch_size=int(X.shape[0]),
                             n_input=X.shape[1],
                             validation_size=validation_size,
                             hidden_layers=hidden_layers,
                             neurons_hidden_layer=neurons_hidden_layer,
                             learning_rate=learning_rate,
                             regularization_parameter=regularization_parameter,
                             kernel_regularizer=kernel_regularizer,
                             repetition=5,
                             loss_function=loss,
                             selection=selection,
                             dropout_value=dropout_value,
                             optimizer="adam",
                             output_activation=output_activation)
information = network.print_info()
build = Net.BuildNetwork(network, X, y)
results_training = build.train_network()
net_model = results_training[0]
error = results_training[1]
print(colored("\nBest error over the repetition:  " + str(error), 'cyan', attrs=['bold']))

y_pred = net_model.predict(X_test)
y_pred = y_pred.reshape(-1,)

if norm:
    if scaler == "m":
        y_pred = y_pred * (max_val - min_val) + min_val
        y_test = y_test * (max_val - min_val) + min_val
    elif scaler == "s":
        y_pred = y_pred *max_val + min_val
        y_test = y_test *max_val + min_val


with open(folder_path+'/previous_error.txt', 'w') as file:
    file.write(str(error))
model_json = net_model.to_json()

if not os.path.exists(folder_path):
    raise NotADirectoryError("The folder " + folder_path + " does not exists")
else:
    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="k")
    plt.xlabel('Actual Data')
    plt.ylabel('Predicted Data')
    # plt.savefig(folder_path + "/Image.png")
    print(colored("\nPrediction error for the current setting:", 'green', attrs=['bold']))
    mean_error = Utils.compute_mean_prediction_error(y_test, y_pred, 2) * 100
    stdv_error = Utils.compute_prediction_error_variance(y_test, y_pred, 2) * 100
    print(str(mean_error) + "%")
    print(str(stdv_error) + "%")

    with open(folder_path + '/MinMax.txt', 'w') as file:
        file.write("Min,Max\n")
        file.write(str(min_val) + "," + str(max_val))
    with open(folder_path + '/Score.txt', 'w') as file:
        file.write("MPE,SPE\n")
        file.write(str(mean_error) + "," + str(stdv_error))
    with open(folder_path + '/Score_reg.txt', 'w') as file:
        file.write("MPE,SPE\n")
        file.write(str(mean_error_reg) + "," + str(std_error_reg))
    if "diff" in keyword:
        with open(folder_path + '/InfoModel.txt', 'w') as file:
            file.write("keyword,variable,samples,level_c,level_f,n_input,scaler\n")
            file.write(str(keyword) + "," +
                       str(variable_name) + "," +
                       str(samples) + "," +
                       str(level_c) + "," +
                       str(level_f) + "," +
                       str(n_input) + "," +
                       str(scaler)
                       )
    else:
        with open(folder_path + '/InfoModel.txt', 'w') as file:
            file.write("keyword,variable,samples,level,n_input,scaler\n")
            file.write(str(keyword) + "," +
                       str(variable_name) + "," +
                       str(samples) + "," +
                       str(level_single) + "," +
                       str(n_input) + "," +
                       str(scaler))

    with open(folder_path + os.sep + "model.json", "w") as json_file:
        json_file.write(model_json)
    # Save weights
    net_model.save_weights(folder_path + os.sep + "model.h5")
    # Save info
    with open(folder_path + os.sep + "Information.csv", "w") as w:
        keys = list(information.keys())
        vals = list(information.values())
        w.write(keys[0])
        for i in range(1, len(keys)):
            w.write("," + keys[i])
        w.write("\n")
        w.write(str(vals[0]))
        for i in range(1, len(vals)):
            w.write("," + str(vals[i]))

