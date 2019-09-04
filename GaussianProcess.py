import matplotlib.pyplot as plt
import UtilsNetwork as Utils
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ExpSineSquared
from matplotlib import rc
import joblib
import os
import sys
import warnings
from termcolor import colored
os.system('color')
warnings.filterwarnings("ignore")


def fit_gaussian_process(X_train, y_train):
    bound = (1e-012, 1000000.0)
    rbf_kernel = RBF(length_scale=1, length_scale_bounds=bound)
    matern_kernel = Matern(length_scale=1.0, length_scale_bounds=bound, nu=0.5)
    matern_kernel_1 = Matern(length_scale=1.0, length_scale_bounds=bound, nu=1.5)
    matern_kernel_2 = Matern(length_scale=1.0, length_scale_bounds=bound, nu=2.5)
    periodic_kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=bound, periodicity_bounds=bound)
    rq_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=bound, alpha_bounds=bound)

    if "_diff" in keyword:
        gp_kernel = matern_kernel_1
    else:
        gp_kernel = matern_kernel_2
    model = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=1500)

    model.fit(X_train, y_train)
    return model


if sys.platform == "win32":
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

keyword = sys.argv[1]
variable_name = sys.argv[2]
samples = int(sys.argv[3])
level_single = sys.argv[4]
level_c = sys.argv[5]
level_f = sys.argv[6]
string_norm = sys.argv[7]
scaler = sys.argv[8]
point = sys.argv[9]

scaler = scaler.replace("'","")
string_norm = string_norm.replace("'","")

print(keyword)
print(variable_name)
print(samples)
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
if "diff" in keyword:
    folder_name = variable_name + "_" + str(level_c) + str(level_f) + "_" + str(samples) + "_" + keyword + "_GP"
else:
    folder_name = variable_name + "_" + str(level_single) + "_" + str(samples) + "_" + keyword + "_GP"

model_path_folder = "./CaseStudies/" + case_folder + "/Models/" + folder_name
if os.path.exists(model_path_folder):
    raise NotADirectoryError("The folder " + folder_name + " already exists")
else:
    os.mkdir(model_path_folder)

if "diff" in keyword:
    X, y, X_test, y_test, min_val, max_val = Utils.get_data_diff(keyword, samples, variable_name, level_c, level_f, n_input,
                                                                 model_path_folder=model_path_folder, normalize=norm, scaler=scaler, point=point)
else:
    X, y, X_test, y_test, min_val, max_val = Utils.get_data(keyword, samples, variable_name, level_single, n_input,
                                                            model_path_folder=model_path_folder, normalize=norm, scaler=scaler, point=point)


mean_error_reg, std_error_reg, model_reg = Utils.linear_regression(keyword, variable_name, X.shape[0], level_c, level_f, level_single, n_input, norm, scaler, point)
filename = model_path_folder + '/model_reg.sav'
joblib.dump(model_reg, filename)

gpr = fit_gaussian_process(X, y)

filename = model_path_folder + '/model_GP.sav'
joblib.dump(gpr, filename)

y_pred, y_std = gpr.predict(X_test, return_std=True)
y_pred = y_pred.reshape(-1,)
y_test = y_test.reshape(-1,)

if norm:
    if scaler == "m" or scaler == "'m'":
        y_pred = y_pred * (max_val - min_val) + min_val
        y_test = y_test * (max_val - min_val) + min_val
    elif scaler == "s" or scaler == "'s'":
        y_pred = y_pred *max_val + min_val
        y_test = y_test *max_val + min_val

print(colored("\nFinal prediction error:", "green", attrs=['bold']))
mean_error = Utils.compute_mean_prediction_error(y_test, y_pred, 2) * 100
variance_error = Utils.compute_prediction_error_variance(y_test, y_pred, 2) * 100
print(str(mean_error) + "%")
print(str(variance_error) + "%")

if sys.platform == "win32":
    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(y_test, y_test, color="k")
    plt.scatter(y_test, y_pred)
    plt.xlabel(r'{Actual Data}')
    plt.ylabel(r'{Predicted Data}')
    plt.savefig(model_path_folder + '/Image.pdf', format='pdf')

with open(model_path_folder + '/MinMax.txt', 'w') as file:
    file.write("Min,Max\n")
    file.write(str(min_val) + "," + str(max_val))
with open(model_path_folder + '/Score.txt', 'w') as file:
    file.write("MPE,SPE\n")
    file.write(str(mean_error) + "," + str(variance_error))
with open(model_path_folder + '/Score_reg.txt', 'w') as file:
    file.write("MPE,SPE\n")
    file.write(str(mean_error_reg) + "," + str(std_error_reg))
if "diff" in keyword:
    with open(model_path_folder + '/InfoModel.txt', 'w') as file:
        file.write("keyword,variable,samples,level_c,level_f,n_input,scaler\n")
        file.write(str(keyword) + "," +
                   str(variable_name) + "," +
                   str(samples) + "," +
                   str(level_c) + "," +
                   str(level_f) + ","+
                   str(n_input) + "," +
                   str(scaler))
else:
    with open(model_path_folder + '/InfoModel.txt', 'w') as file:
        file.write("keyword,variable,samples,level,n_input,scaler\n")
        file.write(str(keyword) + "," +
                   str(variable_name) + "," +
                   str(samples) + "," +
                   str(level_single) + "," +
                   str(n_input) + "," +
                   str(scaler))