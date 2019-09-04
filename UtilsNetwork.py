import tensorflow
import tensorflow.keras as k
import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.linear_model import LinearRegression
from termcolor import colored
import config
import sys
import scipy
from sklearn.model_selection import train_test_split
os.system('color')


#def get_data(keyword, samples, var_name, level, n_input, model_path_folder=None, normalize=True, scaler="m", rs=None):
def get_data(keyword, samples, var_name, level, n_input, model_path_folder=None, normalize=True, scaler="m", point="sobol", rs=None):
    if point != "sobol" and point != "random":
        raise ValueError("check point argument")
    if keyword == "parab":
        dataset = pd.read_csv("./CaseStudies/Parabolic/Data/solution_"+point+"_deltaT_" + str(level) + ".csv", header=0, sep=",")
    elif keyword == "shock":
        dataset = pd.read_csv("./CaseStudies/ShockTube/Data/shock_tube_" + str(level) + ".csv", header=0, sep=",")
    elif keyword == "airf":
        dataset = pd.read_csv("./CaseStudies/Airfoil/Data/airfoil_data_"+str(level)+".csv", header=0, sep=",")
    else:
        raise ValueError("Chose one option between parab, shock and airf")

    if point == "random" and keyword!="parab":
        raise ValueError("Random Point available only for Projectile Motion")

    #print(dataset.head())
    if samples == "all" or samples == "All":
        samples = len(dataset)-1

    if scaler == "m":
        min_val = min(dataset[var_name])
        max_val = max(dataset[var_name])
    elif scaler == "s":
        min_val = dataset[var_name].mean()
        max_val = dataset[var_name].std()
    else:
        raise ValueError("Select one scaler between MinMax (m) and Standard (s)")
    # change here, don't like it
    if normalize:
        if scaler == "m":
            dataset[var_name] = (dataset[var_name] - min_val)/(max_val - min_val)
        elif scaler == "s":
            dataset[var_name] = (dataset[var_name] - min_val)/max_val
    else:
        min_val = 0
        max_val = 1
    #print("Mean: ",dataset[var_name].mean())
    #print("Deviation: ",dataset[var_name].std())
    print(dataset.head())
    loc_var_name = dataset.columns.get_loc(var_name)
    if rs is not None:
        X, X_test, y, y_test = train_test_split(dataset.iloc[:, :n_input].values,dataset.iloc[:, loc_var_name].values,train_size=samples,shuffle=True, random_state=rs)
    else:
        X = dataset.iloc[:samples, :n_input].values
        y = dataset.iloc[:samples, loc_var_name].values
        X_test = dataset.iloc[samples:, :n_input].values
        y_test = dataset.iloc[samples:, loc_var_name].values
    print(X)

    if model_path_folder is not None:
        with open(model_path_folder + '/InfoData.txt', 'w') as file:
            file.write("dev_norm_train,dev_norm_test,dev_train,dev_test,mean_norm_train,mean_norm_test,mean_train,mean_test,\n")
            file.write(str(np.std(y)) + "," +
                       str(np.std(y_test)) + "," +
                       str(np.std(y*(max_val-min_val)+min_val)) + "," +
                       str(np.std(y_test*(max_val-min_val)+min_val)) + "," +
                       str(np.mean(y)) + "," +
                       str(np.mean(y_test)) + "," +
                       str(np.mean(y * (max_val - min_val) + min_val)) + "," +
                       str(np.mean(y_test * (max_val - min_val) + min_val))
                       )

    return X, y, X_test, y_test, min_val, max_val


def get_data_diff(keyword, samples, var_name, level_c, level_f, n_input, model_path_folder=None, normalize=True, scaler ="m", point="sobol", rs=None):
    if point != "sobol" and point !="random":
        raise ValueError("check point argument")
    if keyword == "parab_diff":
        CaseStudy = "Parabolic"
        base = "solution_"+point+"_deltaT_"
    elif keyword == "shock_diff":
        CaseStudy = "ShockTube"
        base = "shock_tube_"
    elif keyword == "airf_diff":
        CaseStudy = "Airfoil"
        base = "airfoil_data_"
    else:
        raise ValueError("Chose one option between parab and shock")

    if point == "random" and keyword != "parab_diff":
        raise ValueError("Random Point available only for Projectile Motion")
    dataset_dt0 = pd.read_csv("./CaseStudies/" + str(CaseStudy) + "/Data/" + base + str(level_c) + ".csv", header=0, sep=",")
    dataset_dt1 = pd.read_csv("./CaseStudies/" + str(CaseStudy) + "/Data/" + base + str(level_f) + ".csv", header=0, sep=",")
    dataset = dataset_dt1
    new_var_name = "diff_" + var_name
    dataset[new_var_name] = (dataset_dt1[var_name] - dataset_dt0[var_name])
    dataset = dataset.drop(var_name, axis=1)
    print(dataset.head())

    if scaler == "m":
        min_val = min(dataset[new_var_name])
        max_val = max(dataset[new_var_name])
    elif scaler == "s":
        min_val = dataset[new_var_name].mean()
        max_val = dataset[new_var_name].std()
    else:
        raise ValueError("Select one scaler between MinMax (m) and Standard (s)")

    if normalize:
        if scaler == "m":
            dataset[new_var_name] = (dataset[new_var_name] - min_val) / (max_val - min_val)
        elif scaler == "s":
            dataset[new_var_name] = (dataset[new_var_name] - min_val) / max_val
    else:
        min_val = 0
        max_val = 1
    #print("Mean: ",dataset[new_var_name].mean())
    #print("Deviation: ",dataset[new_var_name].std())
    # print("min difference:", min_val)
    # print("max difference:", max_val)
    # print("Mean:", dataset[new_var_name].mean())
    # print("Dev:", dataset[new_var_name].std())
    if samples == "all" or samples == "All":
        samples = len(dataset[new_var_name])-1
    loc_var_name = dataset.columns.get_loc(new_var_name)
    if rs is not None:
        X, X_test, y, y_test = train_test_split(dataset.iloc[:, :n_input].values, dataset.iloc[:, loc_var_name].values, train_size=samples, shuffle=True, random_state=rs)
    else:
        X = dataset.iloc[:samples, :n_input].values
        y = dataset.iloc[:samples, loc_var_name].values
        X_test = dataset.iloc[samples:, :n_input].values
        y_test = dataset.iloc[samples:, loc_var_name].values
    #print(X.shape)
    #print(y.shape)

    if model_path_folder is not None:
        with open(model_path_folder + '/InfoData.txt', 'w') as file:
            file.write("dev_norm_train,dev_norm_test,dev_train,dev_test,mean_norm_train,mean_norm_test,mean_train,mean_test,\n")
            file.write(str(np.std(y)) + "," +
                       str(np.std(y_test)) + "," +
                       str(np.std(y*(max_val-min_val)+min_val)) + "," +
                       str(np.std(y_test*(max_val-min_val)+min_val)) + "," +
                       str(np.mean(y)) + "," +
                       str(np.mean(y_test)) + "," +
                       str(np.mean(y * (max_val - min_val) + min_val)) + "," +
                       str(np.mean(y_test * (max_val - min_val) + min_val))
                       )
    return X, y, X_test, y_test, min_val, max_val


'''
elif keyword == "airf_diff":
    CaseStudy = "Airfoil"
    var_name_1 = var_name
    dataset_1 = pd.read_csv("./CaseStudies/" + str(CaseStudy) + "/Data/airfoil_level_" + str(level_c) + ".csv", header=0, sep=",", index_col=0)
    dataset_2 = pd.read_csv("./CaseStudies/" + str(CaseStudy) + "/Data/airfoil_level_" + str(level_f) + ".csv", header=0, sep=",", index_col=0)
    dataset_finest = pd.read_csv("./CaseStudies/" + str(CaseStudy) + "/Data/airfoil_level_4.csv", header=0, sep=",", index_col=0)
    mean_value = np.mean((dataset_finest[var_name]).values ** 2)
    loc_var_name = dataset_1.columns.get_loc(var_name)
    filtered_1 = dataset_1.loc[dataset_2.index]
    filtered_1 = filtered_1.dropna()
    filtered_2 = dataset_2.loc[filtered_1.index]
    filtered_2 = filtered_2.dropna()
    vec_diff = filtered_2.iloc[:, loc_var_name] - filtered_1.iloc[:, loc_var_name]
    relative_var_diff_f = np.var(vec_diff) / np.var(filtered_2.iloc[:, loc_var_name])
    relative_var_diff_c = np.var(vec_diff) / np.var(filtered_1.iloc[:, loc_var_name])
    relative_var_diff_finest = np.var(vec_diff) / np.var(dataset_finest.iloc[:, loc_var_name])
    realtive_var_mean = np.var(vec_diff) / mean_value
    dataset = dataset_2.loc[vec_diff.index]
    dataset = dataset.iloc[:, :6]
    var_name = "diff"
    dataset[var_name] = vec_diff.values

    min_val = min(dataset[var_name])
    max_val = max(dataset[var_name])
    relative_mean_diff = (np.mean((dataset["diff"]).values ** 2) / mean_value) ** (1 / 2)
    print("mean difference:", relative_mean_diff)
    print("min difference:", min_val)
    print("max difference:", max_val)
    print("variances:", relative_var_diff_c, relative_var_diff_f, relative_var_diff_finest, realtive_var_mean)
    dataset[var_name] = (dataset[var_name] - min_val) / (max_val - min_val)

    # with open("Files/AirfoilData/Info_200_new.txt", "a") as file:
    #    file.write("\n" + str(dataset[var_name].var()))
    # with open('./Models/GPModels_SL/MinMax_' + str(samples) + '.txt', 'w') as file:
    #     file.write(str(min_val) + "," + str(max_val))
    X = dataset.iloc[:samples, :6].values
    y = dataset.iloc[:samples, dataset.columns.get_loc(var_name)].values
    X_test = dataset.iloc[samples:, :6].values
    y_test = dataset.iloc[samples:, dataset.columns.get_loc(var_name)].values

    with open('./Files/AirfoilData/Info_' + str(samples) + var_name_1 + '.txt', 'w') as file:
        file.write("Train_Sample,rel_var_finest,mean_relative\n")
        file.write(str(samples)
                   + "," + str(relative_var_diff_finest)
                   + "," + str(relative_mean_diff)
                   )
'''


def load_data(folder_name):
    folder_path = folder_name + os.sep
    info = pd.read_csv(folder_path+"Information.csv", sep=",", header=0)
    optimizer = info.optimizer[0]
    loss = info.loss_function[0]
    learning_rate = info.learning_rate[0]
    if optimizer == "adam":
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=learning_rate)

    with open(folder_path + "model.json") as json_file:
        loaded_model_json = json_file.read()

        loaded_model = k.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(folder_path + "model.h5")

        loaded_model.compile(optimizer=optimizer, loss=loss)

    return loaded_model


def save_model(best_model, information, mean_prediction_error, std_prediction_error, name="Network"):
    i = 0
    folder_name = name
    while True:
        # folder_name = name + "_" + str(i)
        # Check if "Number_0" exists
        if not os.path.exists("Models/"+folder_name):
            os.makedirs("Models/"+folder_name)
            # Save model
            model_json = best_model.to_json()
            with open("Models" + os.sep + folder_name + os.sep + "model.json", "w") as json_file:
                json_file.write(model_json)
            # Save weights
            best_model.save_weights("Models" + os.sep + folder_name + os.sep + "model.h5")
            # Save info
            with open("Models" + os.sep + folder_name + os.sep + "Information.csv", "w") as w:
                keys = list(information.keys())
                vals = list(information.values())
                w.write(keys[0])
                for i in range(1, len(keys)):
                    w.write(","+keys[i])
                w.write("\n")
                w.write(str(vals[0]))
                for i in range(1, len(vals)):
                    w.write("," + str(vals[i]))

            with open("Models" + os.sep + folder_name + os.sep + "Scores.csv", "w") as w:
                w.write("mean_prediction_error" + ":" + str(mean_prediction_error)+"\n")
                w.write("std_prediction_error" + ":" + str(std_prediction_error)+"\n")
            break
        else:
            folder_name = name + "_" + str(i)
            i = i + 1


def compute_mean_prediction_error(data, predicted_data, order):
    base = np.mean(abs(data)**order)
    samples = abs(data-predicted_data)**order
    return (np.mean(samples) / base)**(1.0/order)


def compute_prediction_error_variance(data, predicted_data, order):
    base = np.mean(abs(data) ** order)
    samples = abs(data - predicted_data) ** order

    return np.std(samples) / base


def compute_p_relative_norm(data, predicted_data, order):
    base = np.linalg.norm(data, order)
    samples = np.linalg.norm(data - predicted_data, order)
    return samples / base


def set_model_folder_name(keyword, variable_name, level_c, level_f, level_single, samples):
    if "diff" in keyword:
        folder_name = variable_name + "_" + str(level_c) + str(level_f) + "_" + str(samples) + "_" + keyword
    else:
        folder_name = variable_name + "_" + str(level_single) + "_" + str(samples) + "_" + keyword

    return folder_name


def compute_time(keyword, level_c, level_f, level_single, samples):
    time = 0
    if "parab" in keyword:
        table_time = pd.read_csv("CaseStudies/Parabolic/Data/ComputationalTime.txt", header=0, sep=",")
    elif "airf" in keyword:
        table_time = pd.read_csv("CaseStudies/Airfoil/Data/time_level.txt", header=0, sep=",")
    elif "shock" in keyword:
        table_time = pd.read_csv("CaseStudies/ShockTube/Data/ComputationalTime.txt", header=0, sep=",")
    elif "burg" in keyword:
        table_time = pd.read_csv("CaseStudies/Burger/Data/ComputationalTime.txt", header=0, sep=",")
    else:
        raise ValueError()
    if "_diff" in keyword:
        time_c = table_time["comp_time"].values[level_c]
        time_f = table_time["comp_time"].values[level_f]
        time = (time_c + time_f)*samples
    else:
        time = table_time["comp_time"].values[level_single]*samples
    return time


def call_GaussianProcess(key_word, var_name, sample_coarsest, lev_coarsest, lev_c, lev_f, string_norm, scaler, point):

    arguments = list()
    arguments.append(str(key_word))
    arguments.append(str(var_name))
    arguments.append(str(sample_coarsest))
    arguments.append(str(lev_coarsest))
    arguments.append(str(lev_c))
    arguments.append(str(lev_f))
    arguments.append(str(string_norm))
    arguments.append(str(scaler))
    arguments.append(str(point))

    if sys.platform == "linux" or sys.platform == "linux2":
        string_to_exec = "bsub python3 GaussianProcess.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        os.system(string_to_exec)

    elif sys.platform == "win32":
        python = os.environ['PYTHON36']
        p = subprocess.Popen([python, "GaussianProcess.py"] + arguments)
        p.wait()


def call_NeuralNetwork_cluster(key_word, n_sample, loss_func, folder_path, var_name, lev_c, lev_f, lev_coarsest, string_norm, validation_size, selection_method, scaler, setup, point):

    arguments = list()
    arguments.append(str(key_word))
    arguments.append(str(n_sample))
    arguments.append(str(loss_func))
    for value in setup:
        arguments.append(str(value))
    # arguments.append(str(previous_error))
    arguments.append(str(folder_path))
    arguments.append(str(var_name))
    arguments.append(str(lev_c))
    arguments.append(str(lev_f))
    arguments.append(str(lev_coarsest))
    # arguments.append(str(number_input))
    arguments.append(str(string_norm))
    arguments.append(str(validation_size))
    arguments.append(str(selection_method))
    arguments.append(str(scaler))
    arguments.append(str(point))

    if sys.platform == "linux" or sys.platform == "linux2":
        string_to_exec = "bsub python3 NetworkSingleConf_tesr.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        os.system(string_to_exec)

    elif sys.platform == "win32":
        python = os.environ['PYTHON36']
        p = subprocess.Popen([python, "NetworkSingleConf_tesr.py"] + arguments)
        p.wait()


def linear_regression(keyword, variable_name, sample, level_c, level_f, level_single, n_input, norm, scaler, point):

    if "diff" in keyword:
        X, y, X_test, y_test, min_val, max_val = get_data_diff(keyword, sample, variable_name, level_c, level_f, n_input, normalize=norm, scaler=scaler, point=point)
    else:
        X, y, X_test, y_test, min_val, max_val = get_data(keyword, sample, variable_name, level_single, n_input, normalize=norm, scaler=scaler, point=point)

    reg = LinearRegression().fit(X, y)

    y_pred = reg.predict(X_test)

    y_test = y_test*(max_val - min_val) + min_val
    y_pred = y_pred*(max_val - min_val) + min_val

    mean_error = compute_mean_prediction_error(y_test, y_pred, 2) * 100
    stdv_error = compute_prediction_error_variance(y_test, y_pred, 2) * 100
    print(colored("\nEvaluate linearity data:", "green", attrs=['bold']))
    print(str(mean_error) + "%")
    print(str(stdv_error) + "%")

    return mean_error, stdv_error, reg


def get_network_conf(keyword, variable_name, level_single, level_diff_c, level_diff_f):
    if keyword == "parab":
        param_grid = config.parameter_grid_parab
    elif keyword == "parab_diff":
        param_grid = config.parameter_grid_parab_diff
    elif keyword == "shock":
        param_grid = config.parameter_grid_shock
    elif keyword == "shock_diff":
        param_grid = config.parameter_grid_shock_diff
    elif keyword == "airf":
        if variable_name == "Lift":
            param_grid = config.parameter_grid_airf
        elif variable_name == "Drag":
            param_grid = config.parameter_grid_airf_drag
        else:
            raise ValueError()
    elif keyword == "airf_diff":
        if variable_name == "Lift":
            param_grid = config.parameter_grid_airf_diff
        elif variable_name == "Drag":
            if level_diff_c == 0 and level_diff_f == 1:
                param_grid = config.parameter_grid_airf_diff_drag_01
            elif level_diff_c == 1 and level_diff_f == 2:
                param_grid = config.parameter_grid_airf_diff_drag_12
            elif level_diff_c == 2 and level_diff_f == 3:
                param_grid = config.parameter_grid_airf_diff_drag_23
            elif level_diff_c == 3 and level_diff_f == 4:
                param_grid = config.parameter_grid_airf_diff_drag_34
            elif level_diff_c == 0 and level_diff_f == 2:
                param_grid = config.parameter_grid_airf_diff_drag_02
            elif level_diff_c == 2 and level_diff_f == 4:
                param_grid = config.parameter_grid_airf_diff_drag_24
            elif level_diff_c == 0 and level_diff_f == 4:
                param_grid = config.parameter_grid_airf_diff_drag_04
            elif level_diff_c == 0 and level_diff_f == 3:
                param_grid = config.parameter_grid_airf_diff_drag_03
            else:
                param_grid = config.parameter_grid_airf_diff_drag_01
    else:
        raise ValueError()
    return param_grid


def compute_wasserstein_distance(y, y_pred):
    return scipy.stats.wasserstein_distance(y, y_pred)


def scale_inverse_data(data, scaler, min_val, max_val):
    if scaler == "m":
        data = data * (max_val - min_val) + min_val
    elif scaler == "s":
        data = data * (max_val) + min_val
    return data


def compute_mean_depth(levels):
    depth_mean = 0
    n=0
    for i in range(len(levels)-1):
        depth_mean = depth_mean + (levels[i] - levels[i+1])
        n = n +1
    return depth_mean/n


def ensemble_model(y_models, y_true):

    ensemble = LinearRegression()
    ensemble.fit(y_models, y_true)
    print(ensemble.coef_)
    return ensemble

