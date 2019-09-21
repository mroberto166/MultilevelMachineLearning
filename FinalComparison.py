import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.interpolate import interp1d


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

keyword_list = ["parab", "airf", "airf"]
variable_name_list = ["x_max", "Lift", "Drag"]
point_list = ["random_only_net", "sobol", "sobol"]


for k in range(len(keyword_list)):

    keyword = keyword_list[k]
    variable_name = variable_name_list[k]
    point = point_list[k]

    if "random" in point:
        names = ["MC", "MLMC", "SL2MC", "ML2MC"]
    else:
        names = ["QMC", "MLQMC", "SL2MC", "ML2MC"]
    save = True

    if keyword == "airf":
        case_study = "Airfoil"
    elif keyword == "parab":
        case_study = "Parabolic"
    elif keyword == "shock":
        case_study = "ShockTube"
    else:
        raise ValueError()

    data_base_path = "CaseStudies/"+case_study+"/Data/"
    results_path = "CaseStudies/"+case_study+"/ResultsUQ_"+point+"/"

    # Import files
    MC_solution = pd.read_csv(results_path + "Results_"+variable_name+"_QMC.csv", header=0, sep =",")
    MLMC_solution = pd.read_csv(results_path + "Results_"+variable_name+"_QMLMC.csv", header=0, sep=",")
    MachineLearn_solution_SL = pd.read_csv(results_path + "Results_"+variable_name+"_MLearning_SL.csv", header=0, sep=",")
    MachineLearn_solution_ML = pd.read_csv(results_path + "Results_"+variable_name+"_MLearning_ML.csv", header=0, sep=",")

    error_mean_MC = MC_solution["MSE_mean"].values
    error_var_MC = MC_solution["MSE_std"].values
    error_wass_MC = MC_solution["MSE_wass_dist"].values
    time_MC = MC_solution["Time"].values
    var_error_mean_MC = MC_solution["std_MSE_mean"].values
    var_error_var_MC = MC_solution["std_MSE_std"].values
    var_error_wass_MC = MC_solution["std_MSE_wass"].values

    error_mean_MLMC = MLMC_solution["MSE_mean"].values
    error_var_MLMC = MLMC_solution["MSE_std"].values
    error_wass_MLMC = MLMC_solution["MSE_wass_dist"].values
    time_MLMC = MLMC_solution["Time"].values
    var_error_mean_MLMC = MLMC_solution["std_MSE_mean"].values
    var_error_var_MLMC = MLMC_solution["std_MSE_std"].values
    var_error_wass_MLMC = MLMC_solution["std_MSE_wass"].values

    error_mean_learn_SL = MachineLearn_solution_SL["MSE_mean"].values
    error_var_learn_SL = MachineLearn_solution_SL["MSE_std"].values
    error_wass_SL = MachineLearn_solution_SL["MSE_wass_dist"].values
    time_learn_SL = MachineLearn_solution_SL["Time"].values
    var_error_mean_learn_SL = MachineLearn_solution_SL["std_MSE_mean"].values
    var_error_var_learn_SL = MachineLearn_solution_SL["std_MSE_std"].values
    var_error_wass_SL = MachineLearn_solution_SL["std_MSE_wass"].values

    error_mean_learn_ML = MachineLearn_solution_ML["MSE_mean"].values
    error_var_learn_ML = MachineLearn_solution_ML["MSE_std"].values
    error_wass_ML = MachineLearn_solution_ML["MSE_wass_dist"].values
    time_learn_ML = MachineLearn_solution_ML["Time"].values
    var_error_mean_learn_ML = MachineLearn_solution_ML["std_MSE_mean"].values
    var_error_var_learn_ML = MachineLearn_solution_ML["std_MSE_std"].values
    var_error_wass_ML = MachineLearn_solution_ML["std_MSE_wass"].values

    fig_mean = plt.figure()
    ax_1 = plt.gca()
    ax_1.plot(time_MC, error_mean_MC, color="DarkRed", marker='o', label=names[0], lw=1.5)
    ax_1.plot(time_MLMC, error_mean_MLMC, color="DarkBlue",marker="v", label=names[1], lw=1.5)
    ax_1.plot(time_learn_SL, error_mean_learn_SL, color="C3",marker="D", label=names[2], lw=1.5)
    ax_1.plot(time_learn_ML, error_mean_learn_ML, color="C0",marker="X", label=names[3], lw=1.5)
    ax_1.fill_between(time_MLMC, error_mean_MLMC - var_error_mean_MLMC, error_mean_MLMC + var_error_mean_MLMC, color="DarkBlue", alpha=0.1, interpolate=True, label="__nolegend__")
    ax_1.fill_between(time_MC, error_mean_MC - var_error_mean_MC, error_mean_MC + var_error_mean_MC, color="DarkRed", alpha=0.1, interpolate=True, label="__nolegend__")
    plt.grid(True, which="both",ls=":")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    plt.ylabel(r'Mean Square Error $\mathcal{R}E_E$')
    plt.xlabel(r'Computational Time')
    plt.title("Standard Deviation Estimator")
    plt.title("Mean Estimator")
    plt.legend(loc=3)
    # plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\' + variable_name + '_mean..pdf', format = 'pdf')

    fig_std = plt.figure()
    ax_1 = plt.gca()
    ax_1.plot(time_MC, error_var_MC, color="DarkRed", marker='o', label=names[0], lw=1.5)
    ax_1.plot(time_MLMC, error_var_MLMC, color="DarkBlue",marker="v", label=names[1], lw=1.5)
    ax_1.plot(time_learn_SL, error_var_learn_SL, color="C3",marker="D", label=names[2], lw=1.5)
    ax_1.plot(time_learn_ML, error_var_learn_ML, color="C0",marker="X", label=names[3], lw=1.5)
    ax_1.fill_between(time_MLMC, error_var_MLMC - var_error_var_MLMC, error_var_MLMC + var_error_var_MLMC, color="DarkBlue", alpha=0.1, interpolate=True, label="__nolegend__")
    ax_1.fill_between(time_MC, error_var_MC - var_error_var_MC, error_var_MC + var_error_var_MC, color="DarkRed", alpha=0.1, interpolate=True, label="__nolegend__")
    plt.grid(True, which="both",ls=":")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    plt.ylabel(r'Mean Square Error $\mathcal{R}E_V$')
    plt.xlabel("Computational Time")
    plt.title("Standard Deviation Estimator")
    plt.legend(loc=3)
    # plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\' + variable_name + '_std.pdf', format = 'pdf')

    fig_wass = plt.figure()
    ax_1 = plt.gca()
    ax_1.plot(time_MC, error_wass_MC, color="DarkRed", marker='o', label=names[0], lw=1.5)
    ax_1.plot(time_MLMC, error_wass_MLMC, color="DarkBlue", marker='v', label=names[1], lw=1.5)
    ax_1.plot(time_learn_SL, error_wass_SL, color="C3",marker="D", label=names[2], lw=1.5)
    ax_1.plot(time_learn_ML, error_wass_ML, color="C0",marker="X", label=names[3], lw=1.5)
    ax_1.fill_between(time_MC, error_wass_MC - var_error_wass_MC, error_wass_MC + var_error_wass_MC, color="DarkRed", alpha=0.1, interpolate=True, label="__nolegend__")
    ax_1.fill_between(time_MLMC, error_wass_MLMC - var_error_wass_MLMC, error_wass_MLMC + var_error_wass_MLMC, color="DarkBlue", alpha=0.1, interpolate=True, label="__nolegend__")
    plt.grid(True, which="both",ls=":")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    plt.ylabel(r'Mean Square Error $\mathcal{R}E_W$')
    plt.xlabel("Computational Time")
    plt.title("Wasserstain Distance")
    plt.legend(loc=3)
    # plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\' + variable_name + '_wass.pdf', format = 'pdf')
    plt.savefig('Images/' + variable_name + '_wass.png', dpi=500)

    # Compute speed up
    min_time = (max(min(time_MC), min(time_MLMC), min(time_learn_SL), min(time_learn_ML)))
    max_time = (min(max(time_MC), max(time_MLMC), max(time_learn_SL), max(time_learn_ML)))
    time_interp = np.linspace(min_time, max_time, 1000)

    var = "wass"
    var_vec = ["mean", "var", "wass"]

    for var in var_vec:
        if var == "mean":
            vec_MC = error_mean_MC
            vec_MLMC = error_mean_MLMC
            vec_learn_SL = error_mean_learn_SL
            vec_learn_ML = error_mean_learn_ML
            title = "Mean Estimation"
            name = "Mean"
        elif var == "var":
            vec_MC = error_var_MC
            vec_MLMC = error_var_MLMC
            vec_learn_SL = error_var_learn_SL
            vec_learn_ML = error_var_learn_ML
            title = "Standard Deviation Estimation"
            name = "Dev"
        elif var == "wass":
            vec_MC = error_wass_MC
            vec_MLMC = error_wass_MLMC
            vec_learn_SL = error_wass_SL
            vec_learn_ML = error_wass_ML
            title = "Wasserstein Distance"
            name = "Dist"
        else:
            raise  ValueError(var + " in var_vec not possible(only mean, var, wass)")

        MC = interp1d(time_MC, vec_MC)
        MLMC = interp1d(time_MLMC, vec_MLMC)
        learn_SL = interp1d(time_learn_SL, vec_learn_SL)
        learn_ML = interp1d(time_learn_ML, vec_learn_ML)
        error_interp_MC = MC(time_interp)
        error_interp_MLMC = MLMC(time_interp)
        error_interp_learn_SL = learn_SL(time_interp)
        error_interp_learn_ML = learn_ML(time_interp)

        eff_MC = error_interp_MC/error_interp_learn_ML
        eff_MLMC = error_interp_MLMC/error_interp_learn_ML
        eff_learn_SL = error_interp_learn_SL/error_interp_learn_ML

        print("\n\n")
        print("###########################################")
        print("Observable: ", variable_name)
        print("Statistical Quantity: ", var)
        print("\n")
        print("Average Speedup wrt MC: ", str(round(np.mean(eff_MC),1)))
        print("Max Speedup wrt MC: ", str(round(max(eff_MC),1)))
        print("\n")
        print("Average Speedup wrt MLMC: ", str(round(np.mean(eff_MLMC),1)))
        print("Max Speedup wrt MLMC: ", str(round(max(eff_MLMC),1)))
        print("\n")
        print("Average Speedup wrt SLMC: ", str(round(np.mean(eff_learn_SL),1)))
        print("Max Speedup wrt SLMC: ", str(round(max(eff_learn_SL), 1)))
quit()
