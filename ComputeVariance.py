import UtilsNetwork as Utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('axes', axisbelow=True, labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

keyword = "airf"
var_name = "Drag"
finest=6
keyword_list = ["parab","airf", "airf"]
var_name_list = ["x_max", "Lift", "Drag"]
finest_list = [6,4,4]
colors = ["DarkRed", "DarkBlue", "C0", "C3"]

for j in range(len(keyword_list)):
    keyword = keyword_list[j]
    var_name = var_name_list[j]
    finest = finest_list[j]
    if keyword == "parab":
        xtick_label = [r'$\ell_k=0$', r'$\ell_k=1$', r'$\ell_k=2$', r'$\ell_k=3$', r'$\ell_k=4$', r'$\ell_k=5$', r'$\ell_k=6$']
        variable = "Horizontal range, Projectile Motion"
        inp=7
        finest = 6
        sequences = [
                     [0, 1, 2, 3, 4, 5, 6],
                     [0, 2, 4, 6],
                     [0, 3, 6],
                     [0, 6]
                     ]
        d=1
    else:
        if var_name == "Drag":
            variable = "Drag, Airfoil"
        else:
            variable = "Lift, Airfoil"
        xtick_label = [r'$\ell_k=0$', r'$\ell_k=1$', r'$\ell_k=2$', r'$\ell_k=3$', r'$\ell_k=4$']
        inp=6
        finest = 4
        sequences = [
                 [0, 1, 2, 3, 4],
                 #[0, 2, 3, 4],
                 [0, 2, 4],
                 [0, 4]
                 ]
        d = 2 + 1
    print("######################")
    print(keyword)
    _, y0, _, _, _, _ = Utils.get_data(keyword, "all", var_name, 0,  inp, model_path_folder=None, normalize=False, scaler="m")
    V0 = np.var(y0)
    _, yL, _, _, _, _ = Utils.get_data(keyword, "all", var_name, finest, inp, model_path_folder=None, normalize=False, scaler="m")
    VL = np.var(yL)

    speedup= V0*2**(-finest*d)
    #print("Standard Deviation 0: ", V0)
    plt.figure()
    plt.title(variable)
    for k in range(len(sequences)):
        seq =sequences[k]
        print("#############################")
        print(seq)

        list_var_ratio = [1]
        terms=[0]
        cm = (len(seq)-1)**2/finest
        for i in range(1, len(seq)):
            #print("--------------------------")
            lev_c = seq[i-1]
            lev_f = seq[i]
            _, y, _, _, _, _ = Utils.get_data_diff(keyword+"_diff", "all", var_name, lev_c, lev_f, 7, model_path_folder=None, normalize=False, scaler="m")
            #print("Levels: "+str(lev_c)+", "+str(lev_f))
            #print("variance: ", np.var(y))
            list_var_ratio.append(np.var(y)/V0)
            terms.append(lev_f)
            speedup = speedup + np.var(y)*2**(-(finest - lev_f)*d)
        plt.grid(True, which="both", ls=":")
        plt.plot(terms, list_var_ratio, label=r'$c_{ml} =$ '+ str(round(cm,2)), marker="o", color=colors[k])
        plt.yscale("log")
        plt.ylabel("Variance Ratio")
        plt.xticks(sequences[0], xtick_label)
        speedup = finest*speedup/VL
        print("SPeed up:", 1/speedup)
    plt.legend(loc=3)
    plt.savefig('../Report/Images/variance_ratio_' + var_name + '.pdf', format='pdf')

plt.show()