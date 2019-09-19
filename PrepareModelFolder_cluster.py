import subprocess
import itertools
import os
import sys
import numpy as np
import UtilsNetwork as Utils


if sys.platform == "linux" or sys.platform == "linux2":
    python = "python3"
else:
    python = os.environ['PYTHON36']

keyword = "parab"
variable_name = "x_max"

if keyword == "parab":
    finest_level = 6
elif keyword == "airf":
    finest_level = 4
else:
    raise ValueError("Chose one between parab and airf")

starting_level = 0

loss = "mse"
norm_finest = "true"
scaler_finest = "m"
point = "random"
model = "NET"

sample_0_vec = [256, 512, 1024, 2048]
sample_finest_vec = [4, 8, 16, 32, 64, 92, 128]

setting = itertools.product(sample_0_vec, sample_finest_vec)

for setup in setting:
    print("#########################################################################")
    print("Current setup: ", setup)

    sample_0 = setup[0]
    sample_finest = setup[1]
    exponent = np.log2(sample_0/sample_finest)/finest_level

    level_list = list()
    sample_list = list()
    norm_vec_list = list()
    scaler_vec_list = list()

    #########################################################################
    # Lowest Model Complexity
    if keyword == "parab" or keyword=="shock":
        levels_6 = [6, 0]
        depth = Utils.compute_mean_depth(levels_6)
        samples_6 = [sample_finest, sample_0]
        norms_6 = ["'true'", "'true'"]
        scaler_6 = ["'m'", "'m'"]
    elif keyword == "airf":
        levels_6 = [4, 0]
        samples_6 = [sample_finest, sample_0]
        norms_6 = ["'true'", "'true'"]
        scaler_6 = ["'m'", "'m'"]
    else:
        raise ValueError("Choose one keyword between airf, parab and shock")

    #########################################################################
    # Intermediate Model Complexity 1
    if keyword == "parab" or keyword == "shock":
        levels_3 = [6, 3, 0]
        depth = Utils.compute_mean_depth(levels_3)
        samples_3 = list()
        norms_3 = ["'true'", "'true'", "'true'"]
        scaler_3 = ["'m'", "'m'", "'m'"]
        n_lev_3 = len(levels_3)
        for i in range(n_lev_3-1):
            depth = levels_3[i] - levels_3[i+1]
            samples_3.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_3[i] ))))
        samples_3.append(sample_0)
    elif keyword == "airf":
        levels_3 = [4, 2, 0]
        depth = Utils.compute_mean_depth(levels_3)
        samples_3 = list()
        norms_3 = ["'true'", "'true'", "'true'"]
        scaler_3 = ["'m'", "'m'", "'m'"]
        n_lev_3 = len(levels_3)
        for i in range(n_lev_3-1):
            samples_3.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_3[i]))))
        samples_3.append(sample_0)
    else:
        raise ValueError("Choose one keyword between airf, parab and shock")

    #########################################################################
    # Intermediate Model Complexity 2
    if keyword == "parab" or keyword== "shock":
        levels_2 = [6, 4, 2, 0]
        depth = Utils.compute_mean_depth(levels_2)
        samples_2 = list()
        norms_2 = ["'true'", "'true'", "'true'", "'true'"]
        scaler_2 = ["'m'", "'m'", "'m'", "'m'"]
        n_lev_2 = len(levels_2)
        for i in range(n_lev_2-1):
            samples_2.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_2[i] ))))
        samples_2.append(sample_0)

    elif keyword == "airf":
        levels_2 = [4, 3, 2, 0]
        depth = Utils.compute_mean_depth(levels_2)
        samples_2 = list()
        norms_2 = ["'true'", "'true'", "'true'", "'true'"]
        scaler_2 = ["'m'", "'m'", "'m'", "'m'"]
        n_lev_2 = len(levels_2)
        for i in range(n_lev_2-1):
            samples_2.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_2[i]))))
        samples_2.append(sample_0)

    else:
        raise ValueError("Choose one keyword between airf, parab and shock")

    #########################################################################
    # Highest Model Complexity 2
    if keyword == "parab" or keyword == "shock":
        depth = 1
        levels_1 = [6, 5, 4, 3, 2, 1, 0]
        samples_1 = list()
        norms_1 = ["'true'", "'true'", "'true'", "'true'", "'true'", "'true'"]
        scaler_1 = ["'m'", "'m'", "'m'", "'m'", "'m'", "'m'"]
        n_lev_1 = len(levels_1)
        for i in range(n_lev_1-1):
            samples_1.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_1[i]))))
        samples_1.append(sample_0)
    elif keyword == "airf":
        levels_1 = [4, 3, 2, 1, 0]
        depth = Utils.compute_mean_depth(levels_1)
        samples_1 = list()
        norms_1 = ["'true'", "'true'", "'true'", "'true'", "'true'"]
        scaler_1 = ["'m'", "'m'", "'m'", "'m'", "'m'"]
        n_lev_1 = len(levels_1)
        for i in range(n_lev_1-1):
            samples_1.append(int(sample_finest * 2 ** (exponent * (finest_level - levels_1[i]))))
        samples_1.append(sample_0)
    else:
        raise ValueError("Choose one keyword between airf, parab and shock")

    level_list.append(levels_1)
    level_list.append(levels_2)
    level_list.append(levels_3)
    level_list.append(levels_6)

    sample_list.append(samples_1)
    sample_list.append(samples_2)
    sample_list.append(samples_3)
    sample_list.append(samples_6)

    norm_vec_list.append(norms_1)
    norm_vec_list.append(norms_2)
    norm_vec_list.append(norms_3)
    norm_vec_list.append(norms_6)

    scaler_vec_list.append(scaler_1)
    scaler_vec_list.append(scaler_2)
    scaler_vec_list.append(scaler_3)
    scaler_vec_list.append(scaler_6)

    for i in range(len(sample_list)):
        arguments = list()
        arguments.append(str(level_list[i]))
        arguments.append(str(sample_list[i]))
        arguments.append(str(keyword))
        arguments.append(str(variable_name))
        arguments.append(str(loss))
        arguments.append(str(norm_vec_list[i]))
        arguments.append(str(norm_finest))
        arguments.append(str(scaler_finest))
        arguments.append(str(scaler_vec_list[i]))
        arguments.append(str(point))
        arguments.append(str(model))

        if sys.platform == "linux" or sys.platform == "linux2":
            string_to_exec = "bsub python3 CreateMultiLevModel_cluster.py "
            for arg in arguments:
                string_to_exec = string_to_exec + " '%s'"
            os.system(string_to_exec % (arguments[0],
                                        arguments[1],
                                        arguments[2],
                                        arguments[3],
                                        arguments[4],
                                        arguments[5],
                                        arguments[6],
                                        arguments[7],
                                        arguments[8],
                                        arguments[9],
                                        arguments[10])
                      )
        elif sys.platform == "win32":
            python = os.environ['PYTHON36']
            p = subprocess.Popen([python, "CreateMultiLevModel.py"] + arguments)
            p.wait()















