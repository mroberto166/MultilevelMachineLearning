import numpy as np
import pandas as pd
import UtilsNetwork as Utils
import joblib
import GenerateDataClass as Generator
import os
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

results_UQ = pd.DataFrame(columns=["Time", "MSE_mean", "MSE_std", "MSE_wass_dist", "std_MSE_mean", "std_MSE_std", "std_MSE_wass"])

keyword = "parab"
variable_name = "x_max"
point = "random"
type_point = "Uniform"
modality = "NET"

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
folder_models_SL = "SingleLevelModels_"+ variable_name + "_" + point
path_models_SL = models_base_path + folder_models_SL

reference_solution = pd.read_csv(data_base_path + file_name, header=0)
mean_ref_sol = np.mean(reference_solution[variable_name])
std_ref_sol = np.std(reference_solution[variable_name])
ref_sol = reference_solution[variable_name]
print("***************************************")
print("Reference Solution")
print(mean_ref_sol)
print(std_ref_sol)

N_run = 30

directories_model = [d for d in os.listdir(path_models_SL) if os.path.isdir(os.path.join(path_models_SL, d))]

if os.path.exists(results_path + "/Results_"+variable_name+"_MLearning_SL.csv"):
    old_results = pd.read_csv(results_path + "/Results_"+variable_name+"_MLearning_SL.csv", header=0, sep=",")
else:
    print("File not found")

k = 0
for direc in directories_model:
    print("###############################################################################")
    dir_path = models_base_path + direc
    print(dir_path)
    level_model_info = pd.read_csv(dir_path + "/InfoModel.txt", header=0, sep=",")

    #############################################################
    MSE_mean = 0
    MSE_std = 0
    MSE_wass_dist = 0
    square_sum_mean = 0
    square_sum_std = 0
    square_sum_wass_dist = 0

    if direc == "Lift_4_3_airf_GP":
        model_finest_GP = joblib.load(dir_path + "/model_GP.sav")
    else:
        model_finest_GP = joblib.load(models_base_path + "/" + direc + "_GP/model_GP.sav")
        model_finest_net = Utils.load_data(dir_path)

    scaler_finest = level_model_info["scaler"].values[0]
    sample_finest = level_model_info.samples.values[0]
    level = level_model_info.level.values[0]
    minmax_finest = pd.read_csv(dir_path + "/MinMax.txt", header=0)
    min_val_finest = minmax_finest.Min.values[0]
    max_val_finest = minmax_finest.Max.values[0]

    print(scaler_finest)
    print(sample_finest)

    time = round(Utils.compute_time(keyword,0,0,level, sample_finest), 2)

    print(time)

    if os.path.exists(results_path + "/Results_" + variable_name + "_MLearning_SL.csv"):
        exisiting_time_not_r = old_results["Time"].values.tolist()
        exisiting_time = [round(elem, 2) for elem in exisiting_time_not_r]
        print(exisiting_time)
    else:
        print("File not found")
        exisiting_time = list()

    if time not in exisiting_time:
        # fig = plt.figure()
        list_pred = list()

        for n in range(N_run):
            print("\n====================================================")
            print("N run =:", n)
            X = Generator.generate_collocation_points(n_sample, n_input, type_point).values
            # final_prediction = (model_finest_GP.predict(X) + model_finest_net.predict(X))/2
            final_prediction_1 = model_finest_net.predict(X)
            final_prediction_2 = model_finest_GP.predict(X)
            final_prediction_1 = final_prediction_1.reshape(-1, )
            final_prediction_2 = final_prediction_2.reshape(-1, )

            if modality == "ENS":
                final_prediction = (final_prediction_1 + final_prediction_2)/2
            elif modality == "NET":
                print(modality)
                final_prediction = final_prediction_1
            elif modality == "GP":
                final_prediction = final_prediction_2
            else:
                raise ValueError()

            if direc == "Lift_4_3_airf_GP":
                final_prediction = final_prediction_2
            final_prediction = final_prediction.reshape(-1,)
            final_prediction = Utils.scale_inverse_data(final_prediction, scaler_finest, min_val_finest, max_val_finest)
            list_pred.append(final_prediction)
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
        time = Utils.compute_time(keyword,0,0,level,sample_finest)

        print("MSE for the mean: ", MSE_mean)
        print("MSE for the deviation: ", MSE_std)
        print("MSE for Wasserstain distance: ", MSE_wass_dist)
        print("Deviation MSE for the mean: ", std_MSE_mean)
        print("Deviation MSE for the mean: ", std_MSE_std)
        print("Deviation MSE for the mean: ", std_MSE_wass)
        print("Computational time:", time)
        complexity = 0
        N0 = int(sample_finest)
        Nf = int(sample_finest)
        results_UQ.loc[k] = [time, MSE_mean, MSE_std, MSE_wass_dist, std_MSE_mean, std_MSE_std, std_MSE_wass]
        k = k + 1
        list_pred = np.array(list_pred)
        list_pred = list_pred.reshape(list_pred.shape[0]*list_pred.shape[1],)
        # sns.distplot(list_pred, label="Appoximation:" +str(sample_finest), kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
        # sns.distplot(reference_solution[variable_name], label="Reference:", kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})

        # plt.legend()
    else:
        print("Time already exisisting")

if os.path.exists(results_path + "/Results_" + variable_name + "_MLearning_SL.csv"):
    results_UQ = results_UQ.append(old_results)
results_UQ = results_UQ.sort_values(by=["Time"])
print(results_UQ)


results_UQ.to_csv(results_path + "/Results_"+variable_name+"_MLearning_SL.csv", header=True, index=False)
plt.show()