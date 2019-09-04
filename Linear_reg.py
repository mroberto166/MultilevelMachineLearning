import matplotlib.pyplot as plt
import UtilsNetwork as Utils
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RBF
# from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from matplotlib import rc
import joblib
import os
import sys
import warnings
from termcolor import colored
from sklearn.model_selection import train_test_split
os.system('color')
warnings.filterwarnings("ignore")

keyword = "shock_diff"
variable_name = "pressure"

level_single = 4
level_c = 0
level_f = 6
string_norm = "true"
scaler = "m"

scaler = scaler.replace("'","")
string_norm = string_norm.replace("'","")

print(keyword)
print(variable_name)

print(level_single)
print(level_c)
print(level_f)
print(string_norm)
print(scaler)

# ====================================================
if string_norm == "true" or string_norm == "'true'":
    norm = True
elif string_norm == "false" or string_norm == "'false'":
    norm = False
else:
    raise ValueError("Norm can be 'true' or 'false'")

# ====================================================
if keyword == "parab":
    n_input = 7
    case_folder = "Parabolic"
elif keyword == "parab_diff":
    n_input = 7
    case_folder = "Parabolic"
elif keyword == "shock":
    n_input = 6
    case_folder = "ShockTube"
elif keyword == "shock_diff":
    n_input = 6
    case_folder = "ShockTube"
elif keyword == "airf_diff":
    n_input = 6
    case_folder = "Airfoil"
elif keyword == "airf":
    n_input = 6
    case_folder = "Airfoil"
else:
    raise ValueError("Chose one option between parab and shock")

# ====================================================

for samples in [4, 8, 16, 32, 64, 128, 1000]:
    if "diff" in keyword:
        X, y, _, _, min_val, max_val = Utils.get_data_diff(keyword, samples, variable_name, level_c, level_f, n_input,
                                                           model_path_folder=None, normalize=norm, scaler=scaler)
    else:
        X, y, _, _, min_val, max_val = Utils.get_data(keyword, samples, variable_name, level_single, n_input,
                                                      model_path_folder=None, normalize=norm, scaler=scaler)

    mean_error_reg, std_error_reg, model_reg = Utils.linear_regression(keyword, variable_name, X.shape[0], level_c, level_f, level_single, n_input, norm, scaler=scaler)

    print(samples, mean_error_reg)