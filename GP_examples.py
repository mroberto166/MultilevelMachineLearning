from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from sklearn.gaussian_process import GaussianProcessRegressor


def f(x):
    return -x**3/3 -2**x**2 + 2 + x**4 -x


#matplotlib.rcParams.update({'font.size': 10})

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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
np.random.seed(42)
bound = (1e-012, 1000000.0)
kernel = RBF(length_scale=0.5, length_scale_bounds=bound)
for hyperparameter in kernel.hyperparameters: print(hyperparameter)
params = kernel.get_params()
for key in sorted(params): print("%s : %s" % (key, params[key]))


x_plot = np.linspace(0,2,500)
x_plot_resh = x_plot.reshape(-1,1)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1500, alpha=1e-5)

# Noisy-free observations
fig = plt.figure()
plt.grid(True, which="both", ls=":")
n_r = 3


y_mean, std = gp.predict(x_plot_resh, return_std=True)
plt.plot(x_plot, y_mean, label=r'Mean Value Prediction', lw=1.5)

for i in range(n_r):
    y_i = gp.sample_y(x_plot_resh, random_state=i)
    plt.plot(x_plot, y_i, color="k", ls="--", alpha=0.75)
    y_i = y_i.reshape(-1,)

plt.fill_between(x_plot, y_mean - 2*std, y_mean + 2*std, alpha=0.25, label=r'95\% Confidence Interval', color="grey")
plt.xlabel(r'$y$')
plt.ylabel(r'$f(y)$')
plt.legend(loc=3, prop={'size': 15})
plt.savefig('../Report/Images/gp_'+str(0)+'.pdf', format='pdf')

n_vec = [2, 4, 8, 16]
for n in n_vec:
    np.random.seed(34)
    x = np.random.uniform(0, 2, n)
    y = f(x) #+ 0.1*np.random.uniform(-1,1,n)
    x = x.reshape(-1, 1)
    gp.fit(x,y)
    fig = plt.figure()
    plt.grid(True, which="both", ls=":")

    y_pred, y_std = gp.predict(x_plot_resh, return_std=True)

    plt.scatter(x, y, marker="v", color="DarkBlue", label=r'Observations')
    plt.plot(x_plot, f(x_plot), color="k", ls="-.",label=r'True Function')
    plt.plot(x_plot, y_pred,label=r'Mean Value Prediction', lw=1.5)
    plt.fill_between(x_plot, y_pred - 2*y_std, y_pred + 2*y_std, alpha=0.25, color="grey", label=r'95\% Confidence Interval')
    plt.xlabel(r'$y$')
    plt.ylabel(r'$f(y)$')
    plt.legend(loc=3, prop={'size': 15})
    plt.savefig('../Report/Images/gp_'+str(n)+'.pdf', format='pdf')

# Noisy Observations
eps_vec = [0.01, 0.1, 0.5, 1.0]

for eps in eps_vec:
    np.random.seed(34)
    n_val = 16
    x = np.random.uniform(0, 2, n_val)
    y = f(x) + 0.1*np.random.normal(0, eps, n_val)
    x = x.reshape(-1, 1)
    gp.fit(x, y)
    fig = plt.figure()
    plt.grid(True, which="both", ls=":")

    y_pred, y_std = gp.predict(x_plot_resh, return_std=True)

    plt.scatter(x, y, marker="v", color="DarkBlue", label=r'Observations')
    plt.plot(x_plot, f(x_plot), color="k", ls="-.", label=r'True Function')
    plt.plot(x_plot, y_pred, label=r'Mean Value Prediction', lw=1.5)
    plt.fill_between(x_plot, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.25, color="grey", label=r'95\% Confidence Interval')
    plt.xlabel(r'$y$')
    plt.ylabel(r'$f(y)$')
    plt.legend(loc=3, prop={'size': 15})
    plt.savefig('../Report/Images/gp_' + str(eps).replace(".","") + '.pdf', format='pdf')

