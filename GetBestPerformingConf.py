import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys


keyword = sys.argv[1]
search_folder = sys.argv[2]
name_list = [search_folder]


if keyword == "parab":
    case_study = "Parabolic"
elif keyword == "airf":
    case_study = "Airfoil"
else:
    raise ValueError("Chose one between parab and airf")

for name in name_list:
    print(name)
    folder_name = name
    folder_path_search = "CaseStudies/" + case_study + "/Models/" + folder_name
    directories_model = [d for d in os.listdir(folder_path_search) if os.path.isdir(os.path.join(folder_path_search, d))]

    reg_list = list()
    lr_list = list()
    hl_list = list()
    neur_list = list()

    i = 0
    index_to_el = list()

    directory_min = ""
    directory_min_score = ""

    score_list = list()
    error_list = list()

    min_error = 200
    score_min = 200
    score_error_min = 200

    for directory in directories_model:
        directory_path = folder_path_search + "/" + directory
        with open(directory_path + "/previous_error.txt", "r") as f:
            error = float(f.readlines(0)[0])
        score = pd.read_csv(directory_path + "/Score.txt", header=0)
        if error < min_error:
            directory_min = directory
            min_error = error
            score_error_min = score.MPE.values[0]
        if score.MPE.values[0] < score_min:
            directory_min_score = directory
            score_min = score.MPE.values[0]

        if error < 1:
            try:
                index_to_el.append(i)
                error_list.append(error)
                score_list.append(score.MPE.values[0])
                info_model = pd.read_csv(directory_path + "/Information.csv", header=0)
                reg_list.append(info_model.regularization_parameter.values[0])
                hl_list.append(info_model.hidden_layers.values[0])
                neur_list.append(info_model.neurons_hidden_layer.values[0])
                lr_list.append(info_model.learning_rate.values[0])
            except:
                print("skip")
        i = i+1

    print("Folder corresponding to minimum selection criterion :", directory_min)
    print("Folder corresponding to minimum generalization error:", directory_min_score)
    print("Minimum value of selection criterion: ",min_error)
    print("Minimum value of generalization error corresponding to minimum selection criterion: ", score_error_min)
    print("Minimum value of generalization error: ",score_min)

    norm_score_list = (np.array(score_list) - min(score_list))/(max(score_list)- min(score_list))
    norm_error_list = (np.array(error_list) - min(error_list))/(max(error_list)- min(error_list))

    fig_1 = plt.figure()
    plt.scatter(norm_score_list, norm_error_list)
    plt.savefig("./"+name)

    path_min = folder_path_search + "/" + directory_min
    info = pd.read_csv(path_min + "/Information.csv", sep=",", header=0)
    print("######################################")
    print("Info best performing network")
    print("Height: ",info.neurons_hidden_layer)
    print("Width: ", info.hidden_layers)
    print("Learning rate: ", info.learning_rate)
    print("Regularization type: ", info.kernel_regularizer)
    print("Regularization parameter: ", info.regularization_parameter)


