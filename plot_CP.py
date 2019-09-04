import matplotlib.pyplot as plt
import pandas as pd
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

cp = pd.read_csv("C:\\Users\\rober\Desktop\\Last Semester\\Master Thesis\\AirfoilData\\CP\\CP_folder\\Cp_"+str(1)+".txt", header=None, sep =" ")
x = cp.iloc[:,0]
p = cp.iloc[:,1]
x_sq = cp.iloc[:,0]**2
p_sq = cp.iloc[:,1]**2
n=0
for i in range(1,2000):
	cp = pd.read_csv("C:\\Users\\rober\Desktop\\Last Semester\\Master Thesis\\AirfoilData\\CP\\CP_folder\\Cp_"+str(i)+".txt", header=None, sep =" ")
	x= x + cp.iloc[:,0]
	p =p + cp.iloc[:,1]
	
	p_sq = p_sq + cp.iloc[:,1]**2
	n=n+1
mean_p = p/n
mean_p_sq = p_sq/n
mean_x = x/n
std = (-mean_p**2 + mean_p_sq)**0.5

print(std)
fig = plt.figure()
plt.grid(True, which="both",ls=":")

plt.plot(mean_x, mean_p,c='C0', lw=2, label="Mean Value")
plt.fill_between(mean_x, mean_p-std, mean_p+std, alpha=0.25,
    color="grey",
    label=r'70\% Confidence Interval')
plt.xlabel(r'$x$')
plt.ylabel(r'$-C_p$ (Pressure Coefficient)')
plt.legend(loc=1)

plt.savefig('../Report/Images/mean_cp.pdf', format='pdf')
plt.savefig('C:\\Users\\rober\\Desktop\\LA Presentation\\mean_cp.png', dpi=500)
plt.show()