import UtilsNetwork as Utils
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import itertools
import pandas as pd
import sys
import subprocess
import sobol_seq
import joblib
from scipy.stats import boxcox

model_type = "gp"
print(sys.argv)
norm = sys.argv[1]

norm_name=norm

if norm == "1":
    norm=1
elif norm == "2":
    norm = 2
elif norm == "inf":
    norm = np.inf

sample_vec = [17, 33, 65, 129, 257, 513, 1025]
case_study = "Parabolic"

parameter_grid = {
        "regularization_parameter": [0.000001],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [5],
        "neurons": [10],
        "dropout_value": [0]
    }
setup = list(itertools.product(*parameter_grid.values()))[0]
folder = sys.argv[2]
if sys.argv[3] == "true":
    train = True
else:
    train = False
point = sys.argv[4]
rs = "None"
path = "CaseStudies/"+case_study+"/Models/"+folder

if point == "random":
    N_run = 60
    fl = "NetworkSingleConf_rand.py"
elif point == "sobol":
    N_run = 60
    fl = "NetworkSingleConf_sobol.py"
else:
    raise ValueError()

if train:
    os.mkdir("CaseStudies/"+case_study+"/Models/"+folder)
    for i in range(len(sample_vec)):
        for run in range(N_run):
            print("#########################")
            print("Sample:", sample_vec[i])
            new_path = path + "/Sample_" + str(sample_vec[i]) + "_" + str(run)
            if model_type == "net":
                arguments = list()
                arguments.append(str("parab"))
                arguments.append(str(sample_vec[i]))
                arguments.append(str("mae"))
                for value in setup:
                    arguments.append(str(value))
                arguments.append(str(new_path))
                arguments.append(str("x_max"))
                arguments.append(str(0))
                arguments.append(str(0))
                arguments.append(str(6))
                arguments.append(str("true"))
                arguments.append(str(1))
                arguments.append(str("train_loss"))
                arguments.append(str("m"))
                arguments.append(str(run))
                arguments.append(point)

                if sys.platform == "linux" or sys.platform == "linux2":
                    string_to_exec = "bsub python3 " + fl
                    for arg in arguments:
                        string_to_exec = string_to_exec + " " + arg
                    os.system(string_to_exec)
                elif sys.platform == "win32":
                    python = os.environ['PYTHON36']
                    p = subprocess.Popen([python, fl] + arguments)
                    p.wait()
            else:
                arguments = list()
                arguments.append(str("parab"))
                arguments.append(str("x_max"))
                arguments.append(str(sample_vec[i]))
                arguments.append(str(6))
                arguments.append(str(0))
                arguments.append(str(0))
                arguments.append(str("true"))
                arguments.append(str("m"))
                arguments.append(str(point))
                arguments.append(str(run))
                arguments.append(new_path)

                if sys.platform == "linux" or sys.platform == "linux2":
                    string_to_exec = "bsub python3 GaussianProcess_bound.py "
                    for arg in arguments:
                        string_to_exec = string_to_exec + " " + arg
                    os.system(string_to_exec)

                elif sys.platform == "win32":
                    python = os.environ['PYTHON36']
                    p = subprocess.Popen([python, "GaussianProcess_bound.py"] + arguments)
                    p.wait()

else:
    _, y, _, _, _, _ = Utils.get_data("parab", 3999, "x_max", 6, 7,model_path_folder=None, normalize=False, scaler="m", point=point)

    std = np.std(y)
    print("standard dev ", std)

    mpe_list = list()
    mpe_list_train = list()
    b1_list = list()
    b2_list = list()
    prod_list=list()

    for i in range(len(sample_vec)):
        print("#########################")
        print("Sample:", sample_vec[i])
        mean_MPE = 0
        mean_train = 0
        mean_bound = 0
        mean_bound_2 = 0
        mean_prod = 0
        N_true = 0
        for run in range(N_run):
            #print("Run: ", run)
            new_path = path + "/Sample_"+str(sample_vec[i])+"_"+str(run)
            path_file = new_path +"/Score.txt"
            path_file_train = new_path + "/Score_train.txt"
            with open(path_file, "r") as f:
                lines = f.readlines()
            scores = lines[1].split(",")
            MPE = float(scores[0])
            if not np.isnan(MPE):
                N_true=N_true+1
                mean_MPE = mean_MPE + MPE**2
            else:
                print("not adding")
            #print("MPE:", MPE)

            with open(path_file_train, "r") as f:
                lines = f.readlines()
            scores = lines[1].split(",")
            MPE_train = float(scores[0])
            mean_train = mean_train + MPE_train**2

            if model_type == "net":
                model = Utils.load_data(new_path)
            else:
                model = joblib.load(new_path + "/model_GP.sav")
            minmax = pd.read_csv(new_path + "/MinMax.txt", header=0)
            min_val = minmax.Min.values[0]
            max_val = minmax.Max.values[0]

            if point == "random":
                preds = model.predict(np.random.uniform(0,1,(1000,7)))
            elif point == "sobol":
                preds = model.predict(sobol_seq.i4_sobol_generate(7, 1000))
            else:
                raise ValueError()
            std_app = np.std(preds*(max_val - min_val)+min_val)
            print("True STD: ",std)
            print("Appr STD: ",std_app)

            '''
            
            prod = 1
            for j in range(len(model.layers)):
                if model.layers[j].get_weights():
                    weight_matrix = model.layers[j].get_weights()[0]
                    # for k in range(weight_matrix.shape[1]):
                    #    print(weight_matrix[:,k])
                    #    print(sum(abs(weight_matrix[:,k])))
                    # print(weight_matrix.shape)
                    # print(weight_matrix)
                    norm_mat = np.linalg.norm(weight_matrix, ord=norm, axis=None, keepdims=False)
                    # print(norm_inf)
                    prod = prod * norm_mat
            '''

            #print(prod)
            #minmax = pd.read_csv(new_path + "/MinMax.txt", header=0)
            #min_val = minmax.Min.values[0]
            #max_val = minmax.Max.values[0]

            #preds = model.predict(np.random.uniform(0,1,(500,7)))
            #std = np.std(preds*(max_val - min_val)+min_val)
            #bound = (std + prod)/sample_vec[i]**0.5
            bound_2 = (2*(std+std_app)/sample_vec[i]**0.5 + MPE_train)**2
            #bound_2 = (2 * (std + std) / sample_vec[i] ** 0.5 + MPE_train) ** 2
            #mean_bound = mean_bound + bound
            mean_bound_2 = mean_bound_2 + bound_2
            #mean_prod = mean_prod + prod

        print(N_true)
        mean_MPE = np.sqrt(mean_MPE/N_true)
        mean_train = np.sqrt(mean_train / N_true)
        #mean_bound = mean_bound / N_run
        mean_bound_2 = np.sqrt(mean_bound_2 / N_true)

        mean_prod = mean_prod / N_run

        mpe_list.append(mean_MPE)
        mpe_list_train.append(mean_train)
        #b1_list.append(mean_bound)
        b2_list.append(mean_bound_2)
        prod_list.append(mean_prod)
        print("Mean Test:", mean_MPE)
        print("Mean Train:",mean_train)

    with open("./"+folder+"_"+point+".txt","w") as fi:
        fi.write("Samples,Generaliz_err,Training_err,Bound\n")
        for i in range(len(sample_vec)):
            fi.write(str(sample_vec[i])+","+str(mpe_list[i])+","+str(mpe_list_train[i])+","+str(b2_list[i])+"\n")
    print("Average prod over samples: ",np.mean(prod_list) )
    reg = LinearRegression().fit(np.log10(sample_vec).reshape(-1,1), np.log10(mpe_list).reshape(-1,1))

    x = np.linspace(min(sample_vec), max(sample_vec), 100)
    x = np.log10(x)

    y=reg.predict(x.reshape(-1,1))

    print('Coefficients: \n', reg.coef_)

    plt.scatter(np.log10(sample_vec), np.log10(mpe_list), label="Mean\nGeneralization Error \n"+ str(-reg.coef_[0][0]))
    plt.scatter(np.log10(sample_vec), np.log10(mpe_list_train), label="Mean\nTraining Error")
    plt.scatter(np.log10(sample_vec), np.log10(b2_list),label="Bound\n std/N^0.5")
    plt.plot(x, y)
    plt.legend(loc=0)
    plt.show()

