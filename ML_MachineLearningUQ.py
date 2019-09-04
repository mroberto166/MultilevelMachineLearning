import numpy as np
import pandas as pd
import UtilsNetwork as Utils
from sklearn.externals import joblib
import config
import GenerateDataClass as Generator
import os


def predict(mod_GP, mod_net, method_train, points, scaler_point, min_value, max_value, path):
    print(method_train)
    #print(path)

    y_pred_GP = mod_GP.predict(points).reshape(-1,)
    y_pred_GP = Utils.scale_inverse_data(y_pred_GP, scaler_point, min_value, max_value)

    y_pred_net = mod_net.predict(points).reshape(-1,)
    y_pred_net = Utils.scale_inverse_data(y_pred_net, scaler_point, min_value, max_value)

    y_pred_mods = [y_pred_net, y_pred_GP]
    y_pred_mods = np.array(y_pred_mods).transpose()
    if method_train == "Mean":
        y_predicted = (y_pred_GP + y_pred_net)/2
    elif method_train == "Ensemble":
        ensemb = joblib.load(path + "/model_ens.sav")
        y_predicted = ensemb.predict(y_pred_mods)
    elif method_train == "NET":
        y_predicted = y_pred_net
    elif method_train == "GP":
        y_predicted = y_pred_GP
    else:
        raise ValueError()
    return y_predicted


np.random.seed(42)

models_information_UQ = pd.DataFrame(columns=["complexity", "N0", "Nf", "time", "mean_error", "var_error"])
results_UQ = pd.DataFrame(columns=["Time", "MSE_mean", "MSE_std", "MSE_wass_dist", "std_MSE_mean", "std_MSE_std", "std_MSE_wass"])

keyword = "parab"
variable_name = "x_max"
point="random"

type_point = "Uniform"


if keyword == "airf":
    file_name = "airfoil_data_4.csv"
    file_data_name = "airfoil_data_"
    finest_level = 4
    case_study = "Airfoil"
    n_input = 6
    n_sample = 10000
elif keyword == "parab":
    file_name = "ref_solution_20k.csv"
    file_data_name = "solution_sobol_deltaT_"
    finest_level = 6
    case_study = "Parabolic"
    n_input = 7
    n_sample = 16000
elif keyword == "shock":
    file_name = "shock_tube_8.csv"
    file_data_name = "shock_tube_"
    finest_level = 6
    case_study = "ShockTube"
    n_input = 6
    n_sample = 100000
else:
    raise ValueError()

folder_name = "/ResultsUQ_"+point+"/"
models_base_path = "CaseStudies/"+case_study+"/Models/"
data_base_path = "CaseStudies/"+case_study+"/Data/"
results_path = "CaseStudies/"+case_study+folder_name
folder_models_ML = "MultiLevelModels_"+variable_name+"_"+point
path_models_ML = models_base_path + folder_models_ML

reference_solution = pd.read_csv(data_base_path + file_name, header=0)
mean_ref_sol = np.mean(reference_solution[variable_name])
std_ref_sol = np.std(reference_solution[variable_name])
ref_sol = reference_solution[variable_name]
print("***************************************")
print("Reference Solution")
print(mean_ref_sol)
print(std_ref_sol)

if os.path.exists(results_path + "/Results_"+variable_name+"_MLearning_ML.csv"):
    old_results = pd.read_csv(results_path + "/Results_"+variable_name+"_MLearning_ML.csv", header=0, sep=",")
else:
    print("File not found")

N_run = 30

directories_model = [d for d in os.listdir(path_models_ML) if os.path.isdir(os.path.join(path_models_ML, d))]

k = 0
for direc in directories_model:

    print("###############################################################################")

    dir_path = path_models_ML + "/" + direc

    time = round(pd.read_csv(dir_path + "/Time.txt", header=None, sep=",").values[0, 0],2)

    print(time)

    if os.path.exists(results_path + "/Results_" + variable_name + "_MLearning_ML.csv"):
        exisiting_time_not_r = old_results["Time"].values.tolist()
        exisiting_time = [round(elem, 2) for elem in exisiting_time_not_r]

        print(exisiting_time)
    else:
        print("File not found")
        exisiting_time = list()

    if time not in exisiting_time:
        print(dir_path)
        level_model_info = pd.read_csv(dir_path + "/Info.csv", header=0, sep=",")

        ##########################################################
        MSE_mean = 0
        MSE_std = 0
        MSE_wass_dist = 0
        square_sum_mean = 0
        square_sum_std = 0
        square_sum_wass_dist = 0

        directories_submodel = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

        gaussian_models = list()
        net_models = list()
        scalers = list()
        method_training = list()
        min_values = list()
        max_values = list()
        sub_dir_path_list = list()

        for subdirec in directories_submodel:
            print(subdirec)

            sub_dir_path = dir_path + "/" + subdirec
            # print(sub_dir_path)
            minmax = pd.read_csv(sub_dir_path + "/MinMax.txt", header=0)
            info_model = pd.read_csv(sub_dir_path + "/InfoModel.txt", header=0, sep=",")
            samples = int(info_model.samples.values[0])
            scaler = info_model.scaler.values[0]
            min_val = minmax.Min.values[0]
            max_val = minmax.Max.values[0]

            model_GP = joblib.load(models_base_path + subdirec + "_GP" + "/model_GP.sav")
            model_net = Utils.load_data(models_base_path + subdirec)

            if "diff" in subdirec:
                level_c = info_model.level_c.values[0]
                level_f = info_model.level_f.values[0]
                method = level_model_info.method[(level_model_info.level_c == level_c) & (level_model_info.level_f == level_f)].values[0]
            else:
                level = info_model.level.values[0]
                method = level_model_info.method[(level_model_info.level_c == level) & (level_model_info.level_f == level)].values[0]

            print(method)
            sub_dir_path_list.append(sub_dir_path)
            gaussian_models.append(model_GP)
            net_models.append(model_net)
            method_training.append(method)
            scalers.append(scaler)
            min_values.append(min_val)
            max_values.append(max_val)

        print(sub_dir_path_list)
        print(method_training)

        for n in range(N_run):
            print("\n====================================================")
            print("N run =:", n)
            X = Generator.generate_collocation_points(n_sample, n_input, type_point).values
            final_prediction = np.linspace(0, 0, n_sample).reshape(-1,)

            for i in range(len(method_training)):
                model_GP = gaussian_models[i]
                model_net = net_models[i]
                method = method_training[i]
                scaler = scalers[i]
                min_value = min_values[i]
                max_value = max_values[i]
                current_path = sub_dir_path_list[i]

                final_prediction = final_prediction + predict(model_GP, model_net, method, X, scaler, min_value, max_value, current_path)

            mean_estimator = np.mean(final_prediction)
            std_estimator = np.std(final_prediction)
            wasserstain_dist = Utils.compute_wasserstein_distance(final_prediction, ref_sol)

            MSE_mean = MSE_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 2
            MSE_std = MSE_std + ((std_estimator - std_ref_sol) / std_ref_sol) ** 2
            MSE_wass_dist = MSE_wass_dist + wasserstain_dist ** 2
            square_sum_mean = square_sum_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 4
            square_sum_std = square_sum_std + ((std_estimator - std_ref_sol) / std_ref_sol) ** 4
            square_sum_wass_dist = square_sum_wass_dist + wasserstain_dist ** 4

        std_MSE_mean = np.sqrt((square_sum_mean - MSE_mean ** 2 / N_run) / (N_run - 1))
        std_MSE_wass = np.sqrt((square_sum_wass_dist - MSE_wass_dist ** 2 / N_run) / (N_run - 1))
        std_MSE_std = np.sqrt((square_sum_std - MSE_std ** 2 / N_run) / (N_run - 1))
        MSE_mean = np.sqrt(MSE_mean / N_run)
        MSE_wass_dist = np.sqrt(MSE_wass_dist / N_run)
        MSE_std = np.sqrt(MSE_std / N_run)

        print("MSE for the mean: ", MSE_mean)
        print("MSE for the deviation: ", MSE_std)
        print("MSE for Wasserstain distance: ", MSE_wass_dist)
        print("Deviation MSE for the mean: ", std_MSE_mean)
        print("Deviation MSE for the mean: ", std_MSE_std)
        print("Deviation MSE for the mean: ", std_MSE_wass)
        print("Computational time:", time)

        # Collect Data
        score_model_info = pd.read_csv(dir_path + "/Score.txt", header=0, sep=",")

        depth = score_model_info.depth.values[0]
        layers = score_model_info.n_layer.values[0]
        N0 = int(direc.split("_")[-2])
        Nf = int(direc.split("_")[-1])
        complexity = layers/depth

        models_information_UQ.loc[k] = [complexity, N0, Nf, time, MSE_mean, MSE_std]
        results_UQ.loc[k] = [time, MSE_mean, MSE_std, MSE_wass_dist, std_MSE_mean, std_MSE_std, std_MSE_wass]
        k=k+1
    else:
        print("Time already exisisting")

if os.path.exists(results_path + "/Results_" + variable_name + "_MLearning_ML.csv"):
    results_UQ = results_UQ.append(old_results)
results_UQ = results_UQ.sort_values(by=["Time"])
print(models_information_UQ)
print(results_UQ)


models_information_UQ.to_csv("UQ_info.csv", index=False)
results_UQ.to_csv(results_path + "/Results_"+variable_name+"_MLearning_ML.csv", header=True, index=False)
