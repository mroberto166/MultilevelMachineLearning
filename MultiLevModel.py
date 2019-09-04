import pandas as pd
import matplotlib.pyplot as plt
import UtilsNetwork as Utils
import joblib
import sys
import os
from matplotlib import rc
from termcolor import colored
from sklearn.linear_model import LinearRegression
import numpy as np
from os import listdir
from os.path import isfile, join


os.system('color')

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


model_folder = sys.argv[1]
case_study = sys.argv[2]
point = sys.argv[3]
modality = sys.argv[4]

samples_finest = int(model_folder.split("_")[-1])
print("Samples FInest:", samples_finest)
print("Modality:", modality)

base_path = "CaseStudies/"+case_study+"/Models/"
path_model_folder = base_path + model_folder

directories_model = [d for d in os.listdir(path_model_folder) if os.path.isdir(os.path.join(path_model_folder, d))]
y_pred_diff_list = list()
finest_grid = 0

depth = 0
n_layer = 0
max_error_diff = 0
error_level_0 = 0

y_sol = None
y_0 = None
keyword_finest = None
n_input_finest = None
variable_name_finest = None
model_type_diff = ["Network", "Gaussian", "Mean", "Ensemble"]
model_type_0 = ["Network", "Gaussian", "Mean", "Ensemble"]

sample_eval = 1999
size_validation_ens = 0.1

# 1999 arif


# level_solutions = list()
# single_solutions = list()
model_list = list()
level_list = list()
print(colored("\n\n==============================================================================","green"))
print(colored("Assembling model for " + path_model_folder, "green"))

k = 0
df_final_model = pd.DataFrame(columns=["level_c", "level_f", "samples", "scaler", "method", "MPE"])
for direc in directories_model:


    dir_path = base_path + model_folder + "/" + direc

    if "diff" in direc:
        print("\n====================================================")
        print(direc)
        info_model = pd.read_csv(dir_path+"/InfoModel.txt", header=0, sep=",")

        keyword = info_model.keyword.values[0]
        variable_name = info_model.variable.values[0]
        level_c = info_model.level_c.values[0]
        level_f = info_model.level_f.values[0]
        n_input_diff = info_model.n_input.values[0]
        scaler=info_model.scaler.values[0]
        samples=int(info_model.samples.values[0])
        depth = model_folder.split("_")[-3]
        n_layer = n_layer + 1

        print("Keyword:", keyword)
        print("Variable:", variable_name)
        print("Coarse Level:", level_c)
        print("Fine Level:", level_f)
        print("Scaler:", scaler)
        print("Samples:", samples)

        if level_f > finest_grid:
            finest_grid = level_f


        X, y, _, _, _, _ = Utils.get_data_diff(keyword, "all", variable_name, level_c, level_f, n_input=n_input_diff, model_path_folder=None, normalize=False, point=point)
        sample_validation_ens = int(samples*size_validation_ens)
        if sample_validation_ens == 0:
            sample_validation_ens = 1
        print(sample_validation_ens)
        X_ens = X[samples:samples+sample_validation_ens, :]
        y_ens = y[samples:samples+sample_validation_ens]
        X_sel = X[samples:, :]
        y_sel = y[samples:]
        X = X[samples_finest:sample_eval, :]
        y = y[samples_finest:sample_eval]

        y = y.reshape(-1, )
        y_sel = y_sel.reshape(-1, )
        y_ens = y_ens.reshape(-1, )

        model_GP = joblib.load(base_path + direc + "_GP" + "/model_GP.sav")
        model_net = Utils.load_data(base_path + direc)
        model_reg = joblib.load(base_path + direc + "/model_reg.sav")

        ypred_net_sel = model_net.predict(X_sel)
        ypred_reg_sel = model_reg.predict(X_sel)
        ypred_GP_sel = model_GP.predict(X_sel)

        ypred_net_ens = model_net.predict(X_ens)
        ypred_reg_ens = model_reg.predict(X_ens)
        ypred_GP_ens = model_GP.predict(X_ens)

        ypred_net = model_net.predict(X)
        ypred_reg = model_reg.predict(X)
        ypred_GP = model_GP.predict(X)

        minmax = pd.read_csv(dir_path + "/MinMax.txt", header=0)
        min_val = minmax.Min.values[0]
        max_val = minmax.Max.values[0]

        # CHange for standard scaler
        if scaler == "m":
            ypred_net = ypred_net * (max_val - min_val) + min_val
            ypred_reg = ypred_reg * (max_val - min_val) + min_val
            ypred_GP = ypred_GP * (max_val - min_val) + min_val

            ypred_net_sel = ypred_net_sel * (max_val - min_val) + min_val
            ypred_reg_sel = ypred_reg_sel * (max_val - min_val) + min_val
            ypred_GP_sel = ypred_GP_sel * (max_val - min_val) + min_val

            ypred_net_ens = ypred_net_ens * (max_val - min_val) + min_val
            ypred_reg_ens = ypred_reg_ens * (max_val - min_val) + min_val
            ypred_GP_ens = ypred_GP_ens * (max_val - min_val) + min_val
        elif scaler == "s":
            ypred_net = ypred_net *max_val + min_val
            ypred_reg = ypred_reg *max_val + min_val
            ypred_GP = ypred_GP *max_val + min_val

            ypred_net_sel = ypred_net_sel * max_val + min_val
            ypred_reg_sel = ypred_reg_sel * max_val + min_val
            ypred_GP_sel = ypred_GP_sel * max_val + min_val

            ypred_net_ens = ypred_net_ens * max_val + min_val
            ypred_reg_ens = ypred_reg_ens * max_val + min_val
            ypred_GP_ens = ypred_GP_ens * max_val + min_val

        ypred_net = ypred_net.reshape(-1, )
        ypred_reg = ypred_reg.reshape(-1, )
        ypred_GP = ypred_GP.reshape(-1, )

        ypred_net_sel = ypred_net_sel.reshape(-1, )
        ypred_reg_sel = ypred_reg_sel.reshape(-1, )
        ypred_GP_sel = ypred_GP_sel.reshape(-1, )

        ypred_net_ens = ypred_net_ens.reshape(-1, )
        ypred_reg_ens = ypred_reg_ens.reshape(-1, )
        ypred_GP_ens = ypred_GP_ens.reshape(-1, )

        print("Network:", Utils.compute_mean_prediction_error(y_sel, ypred_net_sel, 2) * 100)
        print("Gaussian:", Utils.compute_mean_prediction_error(y_sel, ypred_GP_sel, 2) * 100)
        print("Regression:", Utils.compute_mean_prediction_error(y_sel, ypred_reg_sel, 2) * 100)

        #ypred_models_diff = list([ypred_net, ypred_reg, ypred_GP])
        #ypred_models_diff.append((ypred_net + ypred_reg + ypred_GP)/3)

        y_pred_mods_sel = [ypred_net_sel, ypred_GP_sel]
        y_pred_mods_sel = np.array(y_pred_mods_sel).transpose()

        y_pred_mods = [ypred_net, ypred_GP]
        y_pred_mods = np.array(y_pred_mods).transpose()

        y_pred_mods_ens = [ypred_net_ens, ypred_GP_ens]
        y_pred_mods_ens = np.array(y_pred_mods_ens).transpose()

        ensemb = Utils.ensemble_model(y_pred_mods_ens, y_ens)
        joblib.dump(ensemb, dir_path + "/model_ens.sav")

        y_ensemble_sel = ensemb.predict(y_pred_mods_sel)
        y_ensemble = ensemb.predict(y_pred_mods)

        #ypred_models_diff = list([ypred_net, ypred_GP])
        ypred_models_diff = list()
        ypred_models_diff_sel = list()
        if modality == "ENS":
            if samples < 500:
                ypred_models_diff.append((ypred_net + ypred_GP) / 2)
                ypred_models_diff_sel.append((ypred_net_sel + ypred_GP_sel) / 2)
                print("Selected Model: Mean")
            else:
                ypred_models_diff.append(y_ensemble)
                ypred_models_diff_sel.append(y_ensemble_sel)
                print("Selected Model: Ensemble")
        elif modality == "NET":
            ypred_models_diff.append(ypred_net)
            ypred_models_diff_sel.append(ypred_net_sel)
            print("Selected Model: Network")
        elif modality == "GP":
            ypred_models_diff.append(ypred_GP)
            ypred_models_diff_sel.append(ypred_GP_sel)
            print("Selected Model: Gaussian Process")
        else:
            raise ValueError("Check Model Input")


        #ypred_models_diff_sel = list([ypred_net_sel, ypred_GP_sel])
        #ypred_models_diff_sel.append((ypred_net_sel + ypred_GP_sel) / 2)
        #ypred_models_diff_sel.append(y_ensemble_sel)

        error_level = 1e+10
        y_diff = None
        model_diff = None
        for i in range(len(ypred_models_diff)):
            pred_sel = ypred_models_diff_sel[i]
            pred = ypred_models_diff[i]
            model = model_type_diff[i]
            error = Utils.compute_mean_prediction_error(y_sel, pred_sel, 2) * 100
            #print("Error : " + model + ": " + str(error))
            if error < error_level:
                error_level = error
                y_diff = pred
                if modality == "ENS":
                    if samples < 500:
                        model_diff = "Mean"
                    else:
                        model_diff = "Ensemble"
                elif modality == "NET":
                    model_diff = "NET"
                elif modality == "GP":
                    model_diff = "GP"
                else:
                    raise ValueError("Check Model Input")

        print("#############################")
        #print("Selected Model: ", model_diff)
        print("Error achieved: ", error_level)
        print("#############################")
        y_pred_diff_list.append(y_diff)
        model_list.append(model_diff)
        level_list.append(level_f)

        df_final_model.loc[k] = [level_c, level_f, samples, scaler, model_diff, error_level]

        if error_level > max_error_diff:
            max_error_diff = error_level

        k = k+1

    else:
        print("\n====================================================")
        print(direc)
        info_model = pd.read_csv(dir_path + "/InfoModel.txt", header=0)
        keyword_0 = info_model.keyword.values[0]
        variable_name = info_model.variable.values[0]
        level = info_model.level.values[0]
        n_input = info_model.n_input.values[0]
        scaler_0 = info_model.scaler.values[0]
        samples_0 = int(info_model.samples.values[0])
        keyword_finest = keyword_0
        n_input_finest = n_input
        variable_name_finest = variable_name

        print("Keyword:", keyword_0)
        print("Variable:", variable_name)
        print("Level:", level)
        print("Number Input:", n_input)
        print("Scaler:", scaler_0)
        print("Samples:", samples_0)

        if level > finest_grid:
            finest_grid = level

        X, y, _, _, _, _ = Utils.get_data(keyword_0, "all", variable_name, level, n_input, normalize=False, point=point)
        sample_validation_ens = int(samples_0*size_validation_ens)
        print(sample_validation_ens)
        X_ens = X[samples_0:samples_0+sample_validation_ens, :]
        y_ens = y[samples_0:samples_0+sample_validation_ens]
        X_sel = X[samples_0:, :]
        y_sel = y[samples_0:]
        X = X[samples_finest:sample_eval, :]
        y = y[samples_finest:sample_eval]

        y = y.reshape(-1, )
        y_sel = y_sel.reshape(-1, )
        y_ens = y_ens.reshape(-1,)

        model_GP = joblib.load(base_path + direc + "_GP" + "/model_GP.sav")
        model_net = Utils.load_data(base_path + direc)
        model_reg = joblib.load(base_path + direc + "/model_reg.sav")

        ypred_0_net = model_net.predict(X)
        ypred_0_reg = model_reg.predict(X)
        ypred_0_GP = model_GP.predict(X)

        ypred_0_net_sel = model_net.predict(X_sel)
        ypred_0_reg_sel = model_reg.predict(X_sel)
        ypred_0_GP_sel = model_GP.predict(X_sel)

        ypred_0_net_ens = model_net.predict(X_ens)
        ypred_0_reg_ens = model_reg.predict(X_ens)
        ypred_0_GP_ens = model_GP.predict(X_ens)

        minmax = pd.read_csv(dir_path + "/MinMax.txt", header=0)
        min_val = minmax.Min.values[0]
        max_val = minmax.Max.values[0]

        if scaler_0 == "m":
            ypred_0_net = ypred_0_net * (max_val - min_val) + min_val
            ypred_0_reg = ypred_0_reg * (max_val - min_val) + min_val
            ypred_0_GP = ypred_0_GP * (max_val - min_val) + min_val

            ypred_0_net_sel = ypred_0_net_sel * (max_val - min_val) + min_val
            ypred_0_reg_sel = ypred_0_reg_sel * (max_val - min_val) + min_val
            ypred_0_GP_sel = ypred_0_GP_sel * (max_val - min_val) + min_val

            ypred_0_net_ens = ypred_0_net_ens * (max_val - min_val) + min_val
            ypred_0_reg_ens = ypred_0_reg_ens * (max_val - min_val) + min_val
            ypred_0_GP_ens = ypred_0_GP_ens * (max_val - min_val) + min_val
        elif scaler_0 == "s":
            ypred_0_net = ypred_0_net * (max_val) + min_val
            ypred_0_reg = ypred_0_reg * (max_val) + min_val
            ypred_0_GP = ypred_0_GP * (max_val) + min_val

            ypred_0_net_sel = ypred_0_net_sel * (max_val) + min_val
            ypred_0_reg_sel = ypred_0_reg_sel * (max_val) + min_val
            ypred_0_GP_sel = ypred_0_GP_sel * (max_val) + min_val

            ypred_0_net_ens = ypred_0_net_sel * (max_val) + min_val
            ypred_0_reg_ens = ypred_0_reg_sel * (max_val) + min_val
            ypred_0_GP_ens = ypred_0_GP_sel * (max_val) + min_val

        ypred_0_reg = ypred_0_reg.reshape(-1, )
        ypred_0_net = ypred_0_net.reshape(-1, )
        ypred_0_GP = ypred_0_GP.reshape(-1, )

        ypred_0_net_sel = ypred_0_net_sel.reshape(-1, )
        ypred_0_reg_sel = ypred_0_reg_sel.reshape(-1, )
        ypred_0_GP_sel = ypred_0_GP_sel.reshape(-1, )

        ypred_0_net_ens = ypred_0_net_ens.reshape(-1, )
        ypred_0_reg_ens = ypred_0_reg_ens.reshape(-1, )
        ypred_0_GP_ens = ypred_0_GP_ens.reshape(-1, )

        print("Network:", Utils.compute_mean_prediction_error(y_sel, ypred_0_net_sel, 2) * 100)
        print("Gaussian:", Utils.compute_mean_prediction_error(y_sel, ypred_0_GP_sel, 2) * 100)
        print("Regression:", Utils.compute_mean_prediction_error(y_sel, ypred_0_reg_sel, 2) * 100)

        y_pred_mods_ens_0 = [ypred_0_net_ens, ypred_0_GP_ens]
        y_pred_mods_ens_0 = np.array(y_pred_mods_ens_0).transpose()

        y_pred_mods_0 = [ypred_0_net, ypred_0_GP]
        y_pred_mods_0 = np.array(y_pred_mods_0).transpose()

        y_pred_mods_sel_0 = [ypred_0_net_sel, ypred_0_GP_sel]
        y_pred_mods_sel_0 = np.array(y_pred_mods_sel_0).transpose()

        ensemb_0 = Utils.ensemble_model(y_pred_mods_ens_0, y_ens)
        joblib.dump(ensemb_0, dir_path+"/model_ens.sav")

        y_ensemble_sel_0 = ensemb_0.predict(y_pred_mods_sel_0)
        y_ensemble_0 = ensemb_0.predict(y_pred_mods_0)

        #ypred_models = list([ypred_0_net, ypred_0_reg, ypred_0_GP])
        #ypred_models.append((ypred_0_net + ypred_0_reg + ypred_0_GP)/3)
        #ypred_models = list([ypred_0_net, ypred_0_GP])
        #ypred_models_sel = list([ypred_0_net_sel, ypred_0_GP_sel])
        ypred_models = list()
        ypred_models_sel = list()
        if modality == "ENS":
            if samples_0 < 500:
                ypred_models.append((ypred_0_net + ypred_0_GP) / 2)
                ypred_models_sel.append((ypred_0_net_sel + ypred_0_GP_sel) / 2)
                print("Selected Model: Mean")
            else:
                ypred_models.append(y_ensemble_0)
                ypred_models_sel.append(y_ensemble_sel_0)
                print("Selected Model: Ensemble")
        elif modality == "NET":
            ypred_models.append(ypred_0_net)
            ypred_models_sel.append(ypred_0_net_sel)
            print("Selected Model: Mean")
        elif modality == "GP":
            ypred_models.append(ypred_0_GP)
            ypred_models_sel.append(ypred_0_GP_sel)
            print("Selected Model: Mean")
        else:
            raise ValueError("Check Model Input")

        error_level_0 = 1e+10
        model_0 = None
        for i in range(len(ypred_models_sel)):
            pred_sel = ypred_models_sel[i]
            pred = ypred_models[i]
            model = model_type_0[i]
            error = Utils.compute_mean_prediction_error(y_sel, pred_sel, 2) * 100
            #print("Error : " + model + ": " + str(error))
            if error < error_level_0:
                error_level_0 = error
                y_sol = pred
                y_0 = pred
                if modality == "ENS":
                    if samples_0 < 500:
                        model_0 = "Mean"
                    else:
                        model_0 = "Ensemble"
                elif modality == "NET":
                    model_0 = "NET"
                elif modality == "GP":
                    model_0 = "GP"
                else:
                    raise ValueError("Check Model Input")

        print("#############################")
        #print("Selected Model: ", model_0)
        print("Error achieved: ", error_level_0)
        print("#############################")
        model_list.append(model_0)
        level_list.append(level)

        df_final_model.loc[k] = [level, level, samples_0, scaler_0, model_0, error_level_0]
        k = k + 1
        # single_solutions.append(ypred_0)
        # level_solutions.append(level)

# level_solutions, single_solutions = zip(*sorted(zip(level_solutions, single_solutions)))
# single_solutions = list(single_solutions)
# level_solutions = list(level_solutions)
# print(single_solutions)
# print(level_solutions)
# print(y_pred_diff_list)
print(df_final_model)
print("\n\n====================================================")
print(samples_finest)
print(finest_grid)
# print(y_sol)
# for i in range(len(single_solutions)-1, 0, -1):
#    print(i)
#    diff = single_solutions[i] - single_solutions[i-1]
#    print(diff)
#    y_sol = y_sol + diff
# print(y_sol)
# print(single_solutions[-1])
# print(y_sol - single_solutions[-1])

for l in y_pred_diff_list:
    y_sol = y_sol + l
y_sol = y_sol.reshape(-1,)
_, y_actual, _, _, min_val, max_val = Utils.get_data(keyword_finest, "all", variable_name_finest, finest_grid, n_input_finest, normalize=False, point=point)
y_actual = y_actual[samples_finest:sample_eval]
fig = plt.figure()
plt.grid(True, which="both", ls=":")
plt.scatter(y_actual, y_sol)
plt.plot(y_actual, y_actual, color="k")
plt.xlabel(r'{Actual Data}')
plt.ylabel(r'{Predicted Data}')
plt.savefig(path_model_folder + "/ImageFinal.pdf", format='pdf')

score_finest = pd.read_csv( path_model_folder + "/Score_fin.txt", header=0, sep=",")
mean_error_finest = score_finest.iloc[0, 0]
std_error_finest = score_finest.iloc[0, 1]

time = pd.read_csv(path_model_folder + "/Time.txt", header=None, sep=",").values[0,0]
time_finest = pd.read_csv(path_model_folder + "/time_finest.txt", header=None, sep=",").values[0,0]
ratio_time = time/time_finest


print(colored("\nPrediction error at coarsest level:", "yellow"))
print(error_level_0)

print(colored("\nMaximum Prediction error for difference function:", "yellow"))
print(max_error_diff)

print(colored("\nBaseline:", "yellow"))
mean_error_bl = Utils.compute_mean_prediction_error(y_actual, y_0, 2) * 100
stdv_error_bl = Utils.compute_prediction_error_variance(y_actual, y_0, 2) * 100
print(str(mean_error_bl) + "%")
print(str(stdv_error_bl) + "%")

print(colored("\nFinal prediction error for multilevel approach:", "yellow"))
mean_error = Utils.compute_mean_prediction_error(y_actual, y_sol, 2) * 100
stdv_error = Utils.compute_prediction_error_variance(y_actual, y_sol, 2) * 100
print(str(mean_error) + "%")
print(str(stdv_error) + "%")

print(colored("\nGain using the multilevel wrt to coarsest grid:", "yellow"))
gain_MPE = mean_error_bl/mean_error
gain_SPE = stdv_error_bl/stdv_error
print(str(gain_MPE))
print(str(gain_SPE))

print(colored("\nGain using the multilevel wrt to finest grid:", "yellow"))
gain_MPE_finest = mean_error_finest/mean_error
gain_SPE_finest = std_error_finest/stdv_error
print(str(gain_MPE_finest))
print(str(gain_SPE_finest))

print(colored("\nRatio MPE coarsest level - finest Level:", "yellow"))
ratio_cf = mean_error/error_level_0
print(ratio_cf)

print(colored("\nModel Goodness:", "yellow"))
G = gain_MPE_finest/ratio_time
print(G)

df_final_model.to_csv(path_model_folder + "/Info.csv", index=False)
with open(path_model_folder + '/Score.txt', 'w') as file:
    file.write("MPE,SPE,base_MPE,base_SPE,gain_MPE_coar,gain_SPE_coar,gain_MPE_fin,gain_SPE_fin,max_MPE_diff,MPE_0,ratio_c_f,depth,n_layer\n")
    file.write(str(mean_error) + "," +
               str(stdv_error) + "," +
               str(mean_error_bl) + "," +
               str(stdv_error_bl) + "," +
               str(gain_MPE) + "," +
               str(gain_SPE) + "," +
               str(gain_MPE_finest) + "," +
               str(gain_SPE_finest) + "," +
               str(max_error_diff) + "," +
               str(error_level_0) + "," +
               str(ratio_cf) + "," +
               str(depth) + "," +
               str(n_layer)
               )
with open(path_model_folder + '/ModelLevelInfo.txt', 'w') as file:
    file.write("level,model\n")
    for i in range(len(model_list)):
        file.write(str(level_list[i]) + "," +
                   str(model_list[i])
                   + "\n")