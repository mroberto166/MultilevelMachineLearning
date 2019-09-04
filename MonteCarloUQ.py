import numpy as np
import pandas as pd
import UtilsNetwork as Utils
import config
import os
import random
import matplotlib.pyplot as plt

random.seed(42)
N_run = 30

keyword = "burg"
variable_name = "y"
point = "random"

if keyword == "airf":
    file_name = "airfoil_data_4.csv"
    file_data_name = "airfoil_data_"
    n_elements = 1999
    finest_level = 4
    case_study = "Airfoil"
    starting = 2
    end = finest_level+1
    M0 = 3
elif keyword == "parab":
    file_name = "ref_solution_20k.csv"
    file_data_name = "solution_"+point+"_deltaT_"
    n_elements = 4000
    finest_level = 6
    case_study = "Parabolic"
    starting = 3
    end = finest_level
    M0=3
elif keyword == "burg":
    n_elements = 7999
    file_name = "Burger_sobol_7.txt"
    file_data_name = "Burger_"+point+"_"
    finest_level = 7
    case_study = "Burger"
    starting = 3
    end = finest_level-1
    M0=1
elif keyword == "shock":
    file_name = "shock_tube_8.csv"
    file_data_name = "shock_tube_"
    n_elements = 16000
    finest_level = 6
    case_study = "ShockTube"
    starting = 3
    end = finest_level
    M0=1
else:
    raise ValueError()

folder_name = "/ResultsUQ_"+point+"/"
data_base_path = "CaseStudies/"+case_study+"/Data/"
results_path = "CaseStudies/"+case_study+folder_name

reference_solution = pd.read_csv(data_base_path + file_name, header=0)
mean_ref_sol = np.mean(reference_solution[variable_name])
std_ref_sol = np.std(reference_solution[variable_name])
ref_sol = reference_solution[variable_name]
print("***************************************")
print("Reference Solution")
print(mean_ref_sol)
print(std_ref_sol)

results = pd.DataFrame(columns=["Time", "MSE_mean", "MSE_std", "MSE_wass_dist", "std_MSE_mean", "std_MSE_std", "std_MSE_wass"])


# Convergence rate of observable (to compute)
s = config.convergence_rate[variable_name][0]
print(s)
#M0=3

levels_list = list()
samples_list = list()
for l in range(starting,end):
    levels_list.append(l)
    print("#######################################")
    MSE_mean = 0
    MSE_std = 0
    MSE_wass_dist = 0
    square_sum_mean = 0
    square_sum_std = 0
    square_sum_wass_dist = 0
    if keyword=="burg":
        data = pd.read_csv(data_base_path + "/" + file_data_name + str(l) + ".txt", sep=",", header=0)
    else:
        data = pd.read_csv(data_base_path + "/" + file_data_name + str(l) + ".csv", sep=",", header=0)
    print(data.head())
    M = int(M0 * 2 ** (2 * s * l))
    samples_list.append(M)
    print("Number of samples:", M)
    print(l)
    for n in range(N_run):
        n_elements = len(data)
        index = random.sample(range(0, n_elements), M)
        vec = data[variable_name].loc[index]
        mean_estimator = np.mean(vec)
        std_estimator = np.std(vec)
        wasserstain_dist = Utils.compute_wasserstein_distance(vec, ref_sol)

        MSE_mean = MSE_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 2
        MSE_std = MSE_std + ((std_estimator - std_ref_sol) / std_ref_sol) ** 2
        MSE_wass_dist = MSE_wass_dist + wasserstain_dist**2
        square_sum_mean = square_sum_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 4
        square_sum_std = square_sum_std + ((std_estimator - std_ref_sol) / std_ref_sol) ** 4
        square_sum_wass_dist = square_sum_wass_dist + wasserstain_dist**4

    std_MSE_mean = np.sqrt((square_sum_mean - MSE_mean ** 2 / N_run) / (N_run - 1))
    std_MSE_wass = np.sqrt((square_sum_wass_dist - MSE_wass_dist ** 2 / N_run) / (N_run - 1))
    std_MSE_std = np.sqrt((square_sum_std - MSE_std ** 2 / N_run) / (N_run - 1))
    MSE_mean = np.sqrt(MSE_mean / N_run)
    MSE_wass_dist = np.sqrt(MSE_wass_dist / N_run)
    MSE_std = np.sqrt(MSE_std / N_run)

    time = Utils.compute_time(keyword, 0, 0 - 1, l, M)
    print("MSE for the mean: ", MSE_mean)
    print("MSE for the deviation: ", MSE_std)
    print("MSE for Wasserstain distance: ", MSE_wass_dist)
    print("Deviation MSE for the mean: ", std_MSE_mean)
    print("Deviation MSE for the mean: ", std_MSE_std)
    print("Deviation MSE for the mean: ", std_MSE_wass)
    print("Computational time:", time)

    results.loc[l] = [time, MSE_mean, MSE_std, MSE_wass_dist, std_MSE_mean, std_MSE_std, std_MSE_wass]

print(results)

results.to_csv(results_path + "/Results_"+variable_name+"_QMC.csv", header=True, index=False)

ax_1 = plt.gca()
ax_1.plot(results.Time,results.MSE_mean)
ax_1.fill_between(results.Time, results.MSE_mean - results.std_MSE_mean, results.MSE_mean + results.std_MSE_mean, color="k", alpha=0.25)
ax_1.set_xscale("log")
ax_1.set_yscale("log")

'''

with open("C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\Report\\MC_data_"+variable_name+".txt","w",encoding='utf-8') as f:
    string = r"\bfseries Resolution  $\ell$  "
    for i in range(len(levels_list)):
        string = string + r'&'+str(levels_list[i])
    string = string + r'\\[3pt]' + "\n"
    f.write(string)
    print(string)
    string = r"\bfseries Samples  $M_\ell$  "
    for i in range(len(samples_list)):
        string = string + r'&' + str(samples_list[i])
    string = string + r'\\[3pt]' + "\n" +r'\midrule'+ "\n"
    f.write(string)
    print(string)
    string = r"\bfseries Runtime  "
    for i in range(len(results["Time"])):
        if variable_name == "Lift" or variable_name == "Drag":
            val_r = 0
            time = int(results["Time"].iloc[i])
        else:
            val_r = 2
            time = results["Time"].iloc[i]
        string = string + r'&' + str(round(time,val_r))
    string = string + r'\\[3pt]' + "\n"
    f.write(string)
    print(string)
plt.show()
'''