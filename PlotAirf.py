import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def x(theta, B):
    return 0.5*(np.abs(np.cos(theta))**B/np.cos(theta) + 1)


def y(theta, P, T, B, E, R, C):
    xx=x(theta, B)
    return T/2*np.abs(np.sin(theta))**B/np.sin(theta)*(1-xx**P) + C*np.sin(xx**E*np.pi) + R*np.sin(2*np.pi*xx)

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


theta = np.linspace(0, 2*np.pi, 1000)

df = pd.read_csv("C:\\Users\\rober\\Desktop\\rae_data.dat", sep="  ", header=None)
x_r=df.iloc[:,0]
y_r=df.iloc[:,1]

B=2
T=0.4
C=0.05
P=1
E=1
R=0

x_vec = x(theta, B)
y_vec= y(theta, P, T, B, E, R, C)

plt.plot(x_r,y_r, color="k", lw=1.5)
plt.grid(True, which="both",ls=":")
plt.fill_between(x_r,y_r,alpha=0.25, color="grey")

plt.xlim(-0.1,1.1)
plt.ylim(-0.3,0.3)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.annotate(r'$\bar{S}_U(x)$', xy=(0, 0), xycoords='data', xytext=(0.42, 0.12), size=15)
plt.annotate(r'$\bar{S}_L(x)$', xy=(0, 0), xycoords='data', xytext=(0.42, -0.12), size=15)
plt.annotate('', xy=(0, 0), xycoords='data', xytext=(1, 0),
            arrowprops=dict(arrowstyle="-", color='C0', lw=1.5,ls=":"))
plt.savefig('../Report/Images/airf.pdf', format='pdf')


case_study="Airfoil"
data_base_path = "CaseStudies/"+case_study+"/Data/"
file_name="airfoil_data_4.csv"
reference_solution_d = pd.read_csv(data_base_path + file_name, header=0)["Lift"]
reference_solution_l = pd.read_csv(data_base_path + file_name, header=0)["Drag"]

plt.figure()
plt.grid(True, which="both",ls=":")
sns.distplot(reference_solution_d, kde=True, hist=True, norm_hist=False)
plt.xlabel(r'Lift Coefficient $C_L$')
plt.savefig('../Report/Images/hist_airf_lift.pdf', format='pdf')
plt.figure()
plt.grid(True, which="both",ls=":")
sns.distplot(reference_solution_l, kde=True, hist=True, norm_hist=False)
plt.xlabel(r'Drag Coefficient $C_D$')
plt.savefig('../Report/Images/hist_airf_drag.pdf', format='pdf')


#plt.title(r'Histogram of IQ: $\mu=17.7$, $\sigma=2.6$')

plt.show()

