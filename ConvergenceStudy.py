import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

var_list = ["x_max", "Lift", "Drag"]
k = 0
norm = "var"

if norm != "2" and norm !="inf"and norm !="var":
    raise ValueError("Choose one between 2 and inf for norm")

convergence_rates = pd.DataFrame(columns=["Variable", "Rate"])

for variable_name in var_list:
    print(variable_name)
    if variable_name == "x_max":
        keyword = "parab"
        case_study = "Parabolic"
        data_base_path = "CaseStudies/" + case_study + "/Data/"
        file_data_name = "solution_sobol_deltaT_"
        n_sample = 2000
        ex_sol = 6
        max_lev = 5
    elif variable_name == "strength":
        keyword = "shock"
        case_study = "ShockTube"
        data_base_path = "CaseStudies/" + case_study + "/Data/"
        file_data_name = "shock_tube_"
        n_sample = 8000
        ex_sol = 8
        max_lev = 5
    elif variable_name == "Lift" or variable_name == "Drag":
        keyword = "airf"
        case_study = "Airfoil"
        data_base_path = "CaseStudies/" + case_study + "/Data/"
        file_data_name = "airfoil_data_"
        n_sample = 1999
        ex_sol = 4
        max_lev = 4
    else:
        raise ValueError("Variable " + variable_name + "not possible (only x_max, Lift, Drag)")

    data_exac = data_i = pd.read_csv(data_base_path+"/"+file_data_name+str(ex_sol)+".csv", sep=",", header=0)[variable_name].values[:n_sample]

    err_vec = list()
    level_list = list()
    for i in range(max_lev):
        data_i = pd.read_csv(data_base_path+"/"+file_data_name+str(i)+".csv", sep=",", header=0)
        # L-infinity norm
        if norm == "inf":
            err = max(abs(data_i[variable_name].values[:n_sample] - data_exac))
        # L-2 norm
        elif norm == "var":
            err = np.std(data_i[variable_name].values[:n_sample] - data_exac)
        else:
            err = np.sqrt(((data_i[variable_name].values[:n_sample]-data_exac)**2).mean())
        err_vec.append(err)
        level_list.append(i)

    fig = plt.figure()
    plt.plot(level_list, np.log(err_vec))

    reg = LinearRegression()
    reg.fit(np.array(level_list).reshape(-1,1), np.log(err_vec).reshape(-1,1))
    print(reg.coef_)
    convergence_rates.loc[k] = [variable_name, -round(reg.coef_[0][0],2)]
    k = k+1

print(convergence_rates)

plt.show()




