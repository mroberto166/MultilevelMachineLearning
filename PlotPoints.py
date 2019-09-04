import sobol_seq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

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

np.random.seed(42)
n_vars = 2
n_samples = 400
collocation_points = sobol_seq.i4_sobol_generate(n_vars, n_samples)
collocation_points_uni = np.random.random((n_samples,n_vars))

x = collocation_points[:,0]
y = collocation_points[:,1]

x_uni = collocation_points_uni[:,0]
y_uni = collocation_points_uni[:,1]

plt.figure()
plt.grid(True, which="both",ls=":")
plt.scatter(x_uni,y_uni)
plt.title("Uniform Random Points")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('../Report/Images/uniform.pdf', format='pdf')

plt.figure()
plt.grid(True, which="both",ls=":")
plt.scatter(x,y, marker="X")
plt.title("Sobol Points")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('../Report/Images/sobol.pdf', format='pdf')
#plt.savefig('C:\\Users\\rober\\Desktop\\LA Presentation\\points.png', dpi=500)
plt.show()


