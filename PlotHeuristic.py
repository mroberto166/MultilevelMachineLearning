import matplotlib.pyplot as plt
import numpy as np
import UtilsNetwork as Utils
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



def func(x, eps, c):
    return x + np.random.normal(c,eps,x.size)

def mpe(eps,c):
    return (eps**4 + c**2)**0.5

np.random.seed(42)
x_vec = np.linspace(0,1, 100)
print(x_vec.size)

y_true = func(x_vec, 0,0)
print("Mean True:",np.mean(y_true))
print("Std True:",np.std(y_true))

plt.figure()
plt.grid(True, which="both",ls=":")
y_1 = func(x_vec, 0.01,0.01)
print("\n")
print("Mean 1:",np.mean(y_1))
print("Std 1:",np.std(y_1))
MPE = (np.mean((y_1 - y_true)**2))**0.5
print(MPE,mpe(0.01,0.01))
plt.scatter(x_vec, y_1, lw=1.5)
plt.plot(x_vec, func(x_vec, 0,0), color="k", ls="--")
plt.xlabel(r'True Values')
plt.ylabel(r'Predicted Values')
plt.savefig('../Report/Images/ideal.pdf', format='pdf')

plt.figure()
plt.grid(True, which="both",ls=":")
y_2 = func(x_vec, 0.2,0.01)
print("\n")
print("Mean 2:",np.mean(y_2))
print("Std 2:",np.std(y_2))
MPE = (np.mean((y_2 - y_true)**2))**0.5
print(MPE,mpe(0.2,0.01))
plt.scatter(x_vec, y_2,lw=1.5)
plt.plot(x_vec, func(x_vec, 0,0), color="k", ls="--")
plt.xlabel(r'True Values')
plt.ylabel(r'Predicted Values')
plt.savefig('../Report/Images/same_mean.pdf', format='pdf')

plt.figure()
plt.grid(True, which="both",ls=":")
y_3 = func(x_vec, 0.01,0.2)
print("\n")
print("Mean 3:",np.mean(y_3))
print("Std 3:",np.std(y_3))
MPE = (np.mean((y_3 - y_true)**2))**0.5
print(MPE,mpe(0.01,0.2))
plt.scatter(x_vec, y_3,lw=1.5)
plt.plot(x_vec, func(x_vec, 0,0), color="k", ls="--")
plt.xlabel(r'True Values')
plt.ylabel(r'Predicted Values')
plt.savefig('../Report/Images/same_std.pdf', format='pdf')
plt.show()