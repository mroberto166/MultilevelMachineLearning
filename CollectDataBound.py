import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('axes', axisbelow=True, labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the tick labels

files = ["TestNet_60.txt"]
initial = ["He, Uniform Inputs"]
des = ["ran"]
title = ["Neural Network, Random Samples"]

for i in range(len(files)):

    data = pd.read_csv("./TestNet/"+files[i], header=0, sep=",")
    samples = data.Samples.values
    GE = data.Generaliz_err.values
    TE = data.Training_err.values
    bounds = data.Bound.values
    compression = bounds/GE

    reg = LinearRegression().fit(np.log10(samples).reshape(-1, 1), np.log10(GE).reshape(-1, 1))

    x = np.linspace(min(samples), max(samples), 100)
    x = np.log10(x)

    y = reg.predict(x.reshape(-1, 1))
    print('Initialization:', initial[i])
    print('Coefficients:', reg.coef_)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(samples, GE, label=r'Generalization Error $\varepsilon_G$', marker="o")
    plt.scatter(samples, TE, label=r'Training Error $\varepsilon_T$', color="DarkBlue", marker= "v")
    plt.plot(samples, bounds, label="Bound",color="DarkRed", marker="d")
    plt.plot(10**(x),10**(y), color="C0",ls=":")
    if i!=0:
        plt.title(title[i])
    plt.legend(loc=1)
    plt.xlabel(r'Training Samples $M$')
    plt.ylabel("Mean Absolute Error")
    plt.savefig('Images/bounds_'+des[i]+'.pdf', format='pdf')

    plt.figure()
    plt.grid(True, which="both", ls="-.")
    plt.plot(samples, compression, marker="o")
    plt.xscale("log")
    plt.xlabel(r'Training Samples $M$')
    plt.ylabel("Compression")
    plt.savefig('Images/compression_'+des[i]+'.pdf', format='pdf')
