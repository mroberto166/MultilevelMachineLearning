import ODE as Solver
import GenerateDataClass as Generator
import numpy as np
import config
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc
import scipy


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

#plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

def make_equal_length(a, new_length):
    old_indices = np.arange(0,len(a))
    new_indices = np.linspace(0,len(a)-1,new_length)
    spl = UnivariateSpline(old_indices,a,k=3,s=0)
    new_array = spl(new_indices)
    return new_array


np.random.seed(42)
delta_t = 0.001
n=1000
means_values = list()
means_values.append(config.density_mean)
means_values.append(config.radius_mean)
means_values.append(config.drag_mean)
means_values.append(config.mass_mean)
means_values.append(config.h_mean)
means_values.append(config.alpha_mean)
means_values.append(config.v0_mean)

epsilon = config.epsilon

collocation_points = Generator.generate_collocation_points(n, 7, "Uniform", param_names=config.parameters_name)
collocation_points = Generator.transform_data(collocation_points, means_values, epsilon)

x_vec = list()
y_vec = list()
time_vec = list()

for index in range(len(collocation_points)):

    params_eq = {
            "gravity": 9.81,
            "density": collocation_points["density"][index],
            "radius": collocation_points["radius"][index],
            "mass": collocation_points["mass"][index],
            "drag_coefficient": collocation_points["drag_coefficient"][index]
        }
    param_initial_conditions = {
        "h": collocation_points["h"][index],
        "alpha": collocation_points["alpha"][index],
        "v0": collocation_points["v0"][index],
    }

    time, x, y = Solver.solve_object_ODE(delta_t, param_initial_conditions, params_eq)
    time_vec.append(time)
    x_vec.append(x)
    y_vec.append(y)

len_max = 0
index_len = 0
for i in range(len(x_vec)):
    if len(x_vec[i]) > len_max:
        len_max = len(x_vec[i])
        index_len = index

print(len_max)
print(index_len)

x_vec_new = list()
y_vec_new = list()
time_vec_new = list()
x_mean = np.linspace(0,0, len_max)
y_mean = np.linspace(0,0, len_max)
time_mean = np.linspace(0,0, len_max)

x_mean_sq = np.linspace(0,0, len_max)
y_mean_sq = np.linspace(0,0, len_max)
fig = plt.figure()
for index in range(len(collocation_points)):
    x_new = make_equal_length(x_vec[index], len_max)
    y_new = make_equal_length(y_vec[index], len_max)
    time_new = make_equal_length(time_vec[index], len_max)

    plt.plot(x_new, y_new)
    x_mean = x_mean + np.array(x_new)
    y_mean = y_mean + np.array(y_new)
    time_mean = time_mean + time_new

    x_mean_sq = x_mean_sq + np.array(x_new)**2
    y_mean_sq = y_mean_sq + np.array(y_new)**2

    x_vec_new.append(x_new)
    y_vec_new.append(y_new)


time_mean = time_mean/n

x_mean = x_mean/n
dev_x = np.sqrt(x_mean_sq/n - x_mean**2)

y_mean = y_mean/n
dev_y = np.sqrt(y_mean_sq/n - y_mean**2)

print(x_mean)
print(y_mean)
print(dev_y)
print(dev_x)

y1 = y_mean -2*dev_y
y2 = y_mean +2*dev_y

x1 = x_mean -2*dev_x
x2 = x_mean +2*dev_x
plt.figure()
plt.grid(True, which="both",ls=":")
plt.plot(x_mean, y_mean, c='C0', lw=2, label="Mean Value")


plt.fill(
    np.append(x1, x2[::-1]),
    np.append(y1, y2[::-1]),
    alpha=0.25,
    color="grey",
    label=r'95\% Confidence Interval'
)
plt.ylabel(r'$x_2$ - coordinate')
plt.xlabel(r'$x_1$ - coordinate')
plt.xlim(0,)
plt.ylim(0,)
plt.legend(loc=1)

'''
x_all = x1 + x2
x_all = np.unique(np.array(x_all))

# interpolate y values on new xarray
y_all = np.empty((len(x_all), 2))
for i,x,y in zip(range(3), [x1,x2], [y1,y2]):
    y_all[:,i] = np.interp(x_all, x, y)

# find out min and max values
ymin = y_all.min(axis=1)
ymax = y_all.max(axis=1)


plt.fill_between(x_all, ymin, ymax, alpha=0.6)

plt.plot(x1,y1 ,'r', label='(AUC = %.2f)')
plt.plot(x2,y2, 'b', label='(AUC = %.2f)')
plt.xlim(0,26)
plt.ylim(0.1,13)
'''
keyword = "parab"
variable_name = "x_max"
if keyword == "airf":
    file_name = "airfoil_data_4.csv"
    file_data_name = "airfoil_data_"
    finest_level = 4
    case_study = "Airfoil"
    n_input = 6
    n_sample = 10000
elif keyword == "parab":
    file_name = "ref_solution_20k.csv"
    file_data_name = "solution_sobol_deltaT_"
    finest_level = 6
    case_study = "Parabolic"
    n_input = 7
    n_sample = 16000
elif keyword == "shock":
    file_name = "shock_tube_8.csv"
    file_data_name = "shock_tube_"
    finest_level = 6
    case_study = "ShockTube"
    n_input = 6
    n_sample = 100000
else:
    raise ValueError()

data_base_path = "CaseStudies/"+case_study+"/Data/"
reference_solution = pd.read_csv(data_base_path + file_name, header=0)[variable_name]
plt.annotate('', xy=(0, 0), xycoords='data', xytext=(reference_solution.mean(), 0),
            arrowprops=dict(arrowstyle="<->", color='DarkRed', lw=2.0))
plt.annotate(r'Horizontal Range $x_{max}$', xy=(0, 0), xycoords='data', xytext=(reference_solution.mean()/4, 0.3), size=12)
plt.savefig('../Report/Images/mean_parab.pdf', format='pdf')
plt.savefig('C:\\Users\\rober\\Desktop\\LA Presentation\\mean_parab.png', dpi=500)
plt.figure()
plt.grid(True, which="both",ls=":")
sns.distplot(reference_solution, kde=True, hist=True, norm_hist=False)
plt.xlabel(r'Horizonatal Range $x_{max}$')
#plt.ylabel('Probability density')
print(round(reference_solution.mean(),1))
print(round(reference_solution.std(),1))
#plt.title(r'Histogram of IQ: $\mu=17.7$, $\sigma=2.6$')
plt.savefig('../Report/Images/hist_'+variable_name+'.pdf', format='pdf')

plt.show()




