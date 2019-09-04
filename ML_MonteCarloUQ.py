import numpy as np
import pandas as pd
import UtilsNetwork as Utils
import config
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
import warnings

warnings.filterwarnings("ignore")


def make_equal_length(a, new_length):
    old_indices = np.arange(0,len(a))
    new_indices = np.linspace(0,len(a)-1,new_length)
    spl = UnivariateSpline(old_indices,a,k=1,s=0)
    new_array = spl(new_indices)
    return new_array


def compute_cdf(vec):
    H, X1 = np.histogram(vec, bins=500, normed=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H) * dx
    return X1[1:], F1


def invert(x_vec, cdf):
    vec = np.linspace(0,1,500)

    idx = np.searchsorted(cdf, vec)

    for i in range(len(idx)):
        if idx[i] == len(x_vec):
            idx[i] = idx[i]-1

    x1 = x_vec[idx - 1]
    x2 = x_vec[idx]
    f1 = cdf[idx - 1]
    f2 = cdf[idx]
    x = x1 + (x2 - x1) / (f2 - f1) * (vec - f1)

    return x


def sum_cdf(X_vec, CDF_vec):
    cdf_tot = np.linspace(0,0, len(X_vec[0]))
    x_tot = np.linspace(0,0, len(X_vec[0]))
    for i in range(len(X_vec)):
        cdf_tot = CDF_vec[i] + cdf_tot
        x_tot = x_tot + X_vec[i]
    return  x_tot, cdf_tot


random.seed(42)
N_run = 30
keyword = "parab"
variable_name = "x_max"
point = "sobol"

if keyword == "airf":
    file_name = "airfoil_data_4.csv"
    file_data_name = "airfoil_data_"
    n_elements = 1999
    finest_level = 4
    case_study = "Airfoil"
    ML_vec = [1, 2, 4, 8, 16, 48]
    finest_level_vec = [3, 4, 4, 4, 4, 4]
    N_levels_vec = [2, 3, 3, 3, 3, 4]
elif keyword == "parab":
    file_name = "ref_solution_20k.csv"
    file_data_name = "solution_"+point+"_deltaT_"
    finest_level = 6
    case_study = "Parabolic"
    ML_vec = [ 2, 4, 8, 16, 32, 64]
    finest_level_vec = [ 4, 4, 4, 4, 4, 5]
    N_levels_vec = [ 5, 5, 4, 4, 4, 4]
elif keyword == "burg":
    file_name = "Burger_sobol_7.txt"
    file_data_name = "Burger_"+point+"_"
    finest_level = 7
    case_study = "Burger"
    ML_vec = [ 1, 2, 4, 8, 32]
    finest_level_vec = [4, 4, 4, 4, 4]
    N_levels_vec = [5, 5, 5, 5, 5, 5]
elif keyword == "shock":
    file_name = "shock_tube_8.csv"
    file_data_name = "shock_tube_"
    n_elements = 16000
    finest_level = 6
    case_study = "ShockTube"
    ML_vec = [1, 2, 4, 8, 16, 32, 64]
    finest_level_vec = [4, 4, 4, 4, 4, 4, 5]
    N_levels_vec = [5, 5, 5, 4, 4, 4, 4]
else:
    raise ValueError()

folder_name = "/ResultsUQ_"+point+"/"
data_base_path = "CaseStudies/"+case_study+"/Data/"
results_path = "CaseStudies/"+case_study+folder_name

results = pd.DataFrame(columns=["Time", "MSE_mean", "MSE_std", "MSE_wass_dist", "std_MSE_mean", "std_MSE_std", "std_MSE_wass"])
reference_solution = pd.read_csv(data_base_path + file_name, header=0)
mean_ref_sol = np.mean(reference_solution[variable_name])
std_ref_sol = np.std(reference_solution[variable_name])
ref_sol = reference_solution[variable_name]
print("***************************************")
print("Reference Solution")
print(mean_ref_sol)
print(std_ref_sol)

# Convergence rate of observable (to compute)
s = config.convergence_rate[variable_name][0]
print(s)

k = 0
for i in range(len(ML_vec)):
    print("#######################################")
    ML = ML_vec[i]
    finest_level = finest_level_vec[i]
    N_levels = N_levels_vec[i]
    MSE_mean = 0
    MSE_std = 0
    MSE_wass_dist = 0
    square_sum_mean = 0
    square_sum_std = 0
    square_sum_wass_dist = 0
    # fig = plt.figure()
    l_pred = list()
    starting_level = finest_level+1-N_levels
    for n in range(N_run):
        mean_estimator = 0
        std_estimator = 0
        # list_pred = list()
        time = 0

        M0 = int(ML * (2 ** (2 * s * (N_levels - 1))))

        x_list = list()
        cdf_vec = list()
        for lev in range(N_levels-1, 0, -1):
            l = starting_level + lev

            Ml = int(ML*(2**(2*s*(N_levels-1-lev))))
            # print(lev, l, Ml)

            if keyword =="burg":
                data_i = pd.read_csv(data_base_path + "/" + file_data_name + str(l) + ".txt", sep=",", header=0)
                data_i_minus = pd.read_csv(data_base_path + "/" + file_data_name + str(l - 1) + ".txt", sep=",", header=0)
            else:
                data_i = pd.read_csv(data_base_path + "/" + file_data_name + str(l) + ".csv", sep=",", header=0)
                data_i_minus = pd.read_csv(data_base_path + "/" + file_data_name + str(l - 1) + ".csv", sep=",", header=0)

            n_elements = len(data_i)
            index = random.sample(range(0, n_elements), Ml)
            vec_i = data_i[variable_name].loc[index]
            vec_i_minus = data_i_minus[variable_name].loc[index]
            mean_estimator = mean_estimator + np.mean(vec_i - vec_i_minus)
            std_estimator = std_estimator + np.std(vec_i) - np.std(vec_i_minus)

            X_diff, cdf_diff = compute_cdf(vec_i)
            X_diff_minus, cdf_diff_minus = compute_cdf(vec_i_minus)
            cdf_diff = cdf_diff - cdf_diff_minus
            X_diff = X_diff - X_diff_minus
            X_diff = make_equal_length(X_diff, M0)
            cdf_diff = make_equal_length(cdf_diff, M0)
            x_list.append(X_diff)
            cdf_vec.append(cdf_diff)

            time = time + Utils.compute_time(keyword+"_diff",l,l-1,0,Ml)
            # print(time)
            # list_pred = list_pred + vec_i.tolist()
            # l_pred = l_pred + vec_i.tolist()

        if keyword == "burg":
            data_0 = pd.read_csv(data_base_path + "/" + file_data_name + str(finest_level - N_levels + 1) + ".txt", sep=",", header=0)
        else:
            data_0 = pd.read_csv(data_base_path + "/" + file_data_name + str(finest_level - N_levels + 1) + ".csv", sep=",", header=0)
        n_elements = len(data_0)
        index = random.sample(range(0, n_elements), M0)
        vec_0 = data_0[variable_name].loc[index]

        X_0, cdf_0 = compute_cdf(vec_0)
        X_0 = make_equal_length(X_0, M0)
        cdf_0 = make_equal_length(cdf_0, M0)
        x_list.append(X_0)
        cdf_vec.append(cdf_0)

        x_total, cdf_total = sum_cdf(x_list, cdf_vec)
        samp = invert(x_total, cdf_total)

        mean_estimator = mean_estimator + np.mean(vec_0)
        std_estimator = std_estimator + np.std(vec_0)
        wasserstain_dist = Utils.compute_wasserstein_distance(samp, ref_sol)
        # print(wasserstain_dist)
        time = time + Utils.compute_time(keyword, 0, 0, finest_level - N_levels+1, M0)

        MSE_mean = MSE_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 2
        MSE_std = MSE_std + ((std_estimator - std_ref_sol) / std_ref_sol) ** 2
        MSE_wass_dist = MSE_wass_dist + wasserstain_dist ** 2
        square_sum_mean = square_sum_mean + ((mean_estimator - mean_ref_sol) / mean_ref_sol) ** 4
        square_sum_std = square_sum_std + ((std_estimator - std_ref_sol)/std_ref_sol) ** 4
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
    # sns.distplot(l_pred, label="Appoximation:" + str(ML), kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
    # sns.distplot(reference_solution[variable_name], label="Reference:", kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
    # plt.legend()
    results.loc[k] = [time, MSE_mean, MSE_std, MSE_wass_dist, std_MSE_mean, std_MSE_std, std_MSE_wass]
    k = k + 1

print(results)

results.to_csv(results_path + "/Results_"+variable_name+"_QMLMC.csv", header=True, index=False)


with open("C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\Report\\MLMC_data_"+variable_name+".txt","w",encoding='utf-8') as f:
    string = r"& &\bfseries Finest Resolution $\ell_L$ &  "
    for i in range(len(finest_level_vec)):
        string = string + r'&'+str(finest_level_vec[i])
    string = string + r'\\' + "\n"
    f.write(string)
    print(string)

    string = r"& & \bfseries Number of levels $n_\ell$ & "
    for i in range(len(N_levels_vec)):
        string = string + r'&' + str(N_levels_vec[i])
    string = string + r'\\' + "\n"
    f.write(string)
    print(string)

    string = r"& &\bfseries Samples $M_L$  & "
    for i in range(len(ML_vec)):
        string = string + r'&' + str(ML_vec[i])
    string = string + r'\\'  + "\n"
    f.write(string)
    print(string)

'''
string = r"& & \bfseries Runtime  &"
    for i in range(len(results["Time"])):
        if variable_name == "Lift" or variable_name == "Drag":
            val_r = 0
            time = int(results["Time"].iloc[i])
        else:
            val_r = 2
            time = results["Time"].iloc[i]
        string = string + r'&' + str(round(time,val_r))
    string = string + r'\\' + "\n"
    f.write(string)
    print(string)
'''

plt.show()

