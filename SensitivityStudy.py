import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import seaborn as sns
import warnings
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('axes', axisbelow=True, labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

save = True
keyword_list = ["parab", "airf", "airf"]
variable_name_list = ["x_max", "Lift", "Drag"]
point_list = ["random_only_net", "sobol", "sobol"]

keyword_list = ["parab"]
variable_name_list = ["x_max"]
point_list = ["random_only_net"]

for k in range(len(keyword_list)):
    keyword = keyword_list[k]
    variable_name = variable_name_list[k]
    point = point_list[k]
    print("\n\n############################################")
    print(keyword, variable_name)
    print("############################################\n")

    if keyword == "airf":
        case_study = "Airfoil"
        finest_level = 4
    elif keyword == "parab":
        case_study = "Parabolic"
        finest_level = 6
    else:
        raise ValueError()

    base_path = "CaseStudies/"+case_study+"/Models/Sensitivity_"+variable_name+"_"+point
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    time_fin_list = list()
    score_fin_list = list()
    score_fin_SPE_list = list()
    i = 0
    sensitivity_df = pd.DataFrame()
    model_df = pd.DataFrame()

    for dir in directories_model:
        if "Depth_"in dir:
            dir_path = base_path + os.sep + dir
            dir_el = dir.split("_")
            N0 = int(dir_el[-2])
            Nf = int(dir_el[-1])

            score = pd.read_csv(dir_path + "/Score.txt", header=0, sep=",")
            time = pd.read_csv(dir_path + "/Time.txt", header=None, sep=",")
            time_finest = pd.read_csv(dir_path + "/time_finest.txt", header=None, sep=",").values[0]
            score_finest = pd.read_csv(dir_path + "/Score_fin.txt", header=0, sep=",")["MPE"].values[0]
            SPE_finest = pd.read_csv(dir_path + "/Score_fin.txt", header=0, sep=",")["SPE"].values[0]
            models = pd.read_csv(dir_path + "/ModelLevelInfo.txt", header=0, sep=",")
            df = score
            df["Time"] = time.values
            df["Time_Finest"] = time_finest
            df["Time Loss"] = time.values/time_finest
            # df["MPE Ratio"] = df["MPE"]/score_base
            # df["Time Ratio"] = df["Time"]/time_base
            df["N0"] = N0
            df["Nf"] = Nf
            df["Model Complexity"] = round(df.n_layer.values[0]/df.depth.values[0],2)
            df["Model Goodness MPE"] = df["gain_MPE_fin"]/df["Time Loss"]
            df["Model Goodness SPE"] = df["gain_SPE_fin"] / df["Time Loss"]
            df["Efficiency"] = np.log(df["MPE"]) *np.log(df["Time"])

            #df["Model Goodness MPE Base"] = 1/(df["MPE Ratio"] * df["Time Ratio"])
            # print(df)
            if i == 0:
                sensitivity_df = df
                model_df = models
            else:
                sensitivity_df = sensitivity_df.append(df)
                model_df = model_df.append(models)

            i = i+1
            score_fin_list.append(score_finest)
            score_fin_SPE_list.append(SPE_finest)
            time_fin_list.append(time_finest)

    sensitivity_df = sensitivity_df.reset_index(drop=True)
    model_df = model_df.reset_index(drop=True)
    # sensitivity_df = sensitivity_df[sensitivity_df.index != sensitivity_df['ratio_c_f'].idxmax()]
    # sensitivity_df = sensitivity_df.reset_index(drop=True)
    sensitivity_df = sensitivity_df.drop("n_layer", axis=1)
    sensitivity_df = sensitivity_df.drop("depth", axis=1)
    # sensitivity_df = sensitivity_df.drop("gain_MPE", axis=1)
    # sensitivity_df = sensitivity_df.drop(" gain_SPE", axis=1)
    sensitivity_df = sensitivity_df.drop("MPE_0", axis=1)

    sensitivity_df = sensitivity_df.loc[(sensitivity_df["Model Goodness MPE"] < 40)]

    # Prepare diagram efficiency vs time (per time intervals)
    N_int = 5
    min_time = np.log10(sensitivity_df["Time"]).min()
    max_time = np.log10(sensitivity_df["Time"]).max()
    span_time = (max_time - min_time)/N_int

    min_MPE_list_per_time = list()
    max_MPE_list_per_time = list()
    ave_MPE_list_per_time = list()
    std_MPE_list_per_time = list()
    time_list = list()

    for i in range(0, N_int):
        index_list_i = sensitivity_df.index[(np.log10(sensitivity_df["Time"]) < min_time + (i+1)*span_time) & (np.log10(sensitivity_df["Time"]) >= min_time + (i)*span_time)]
        new_df = sensitivity_df.ix[index_list_i]
        time_list.append(new_df["Time"].mean())

        min_MPE_list_per_time.append(new_df["Model Goodness MPE"].min())
        max_MPE_list_per_time.append(new_df["Model Goodness MPE"].max())
        ave_MPE_list_per_time.append(new_df["Model Goodness MPE"].mean())
        std_MPE_list_per_time.append(new_df["Model Goodness MPE"].std())


    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    ax.set_axisbelow(True)
    plt.scatter(time_list, max_MPE_list_per_time,marker="v", color="C0", label=r'Max', s=50)
    plt.plot(time_list, ave_MPE_list_per_time, marker="*", color="C0", label=r'Mean', ls="--", markersize=10)
    plt.xscale('log')
    plt.xlabel(r'Computational Time')
    plt.ylabel(r'Accuracy Speed Up $G$')
    plt.legend()
    plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\error_cost_' + variable_name + '.pdf', format='pdf')

    # Prepare diagram efficiency vs error (per erro intervals)
    N_int = 5
    min_MPE = np.log10(sensitivity_df["MPE"]).min()
    max_MPE = np.log10(sensitivity_df["MPE"]).max()
    span_MPE = (max_MPE - min_MPE)/N_int

    min_MPE_list_per_MPE = list()
    max_MPE_list_per_MPE = list()
    ave_MPE_list_per_MPE = list()
    std_MPE_list_per_MPE = list()
    MPE_list_ = list()

    for i in range(0, N_int):

        index_list_i = sensitivity_df.index[(np.log10(sensitivity_df["MPE"]) < min_MPE + (i+1)*span_MPE) & (np.log10(sensitivity_df["MPE"]) >= min_MPE + (i)*span_MPE)]
        new_df = sensitivity_df.ix[index_list_i]
        MPE_list_.append(new_df["MPE"].mean())
        min_MPE_list_per_MPE.append(new_df["Model Goodness MPE"].min())
        max_MPE_list_per_MPE.append(new_df["Model Goodness MPE"].max())
        ave_MPE_list_per_MPE.append(new_df["Model Goodness MPE"].mean())
        std_MPE_list_per_MPE.append(new_df["Model Goodness MPE"].std())


    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    ax.set_axisbelow(True)
    plt.scatter(MPE_list_, max_MPE_list_per_MPE,marker="v", color="C0", label=r'Max', s=50)
    plt.plot(MPE_list_, ave_MPE_list_per_MPE, marker="*", color="C0", label=r'Mean', ls="--", markersize=10)
    plt.xscale('log')
    x_value=[str(x)+ r'\%' for x in ax.get_xticks()]
    ax.xaxis.set_major_formatter(PercentFormatter())
    plt.xlabel(r'Multilevel MPE $\varepsilon_{ml}$')
    plt.ylabel(r'Accuracy Speed Up $G$')
    plt.legend()


    # split dataframe into 3 df for each parameter of interest
    print("=======================================================")
    print("Number of sample at finest grid")
    Nf_vec = sensitivity_df["Nf"].values
    Nf_vec = list(set(Nf_vec))
    Nf_vec.sort()
    df_Nf_list = list()
    for sample_f in Nf_vec:
        index_list_i = sensitivity_df.index[sensitivity_df.Nf == sample_f]
        new_df = sensitivity_df.ix[index_list_i]
        df_Nf_list.append(new_df)

    print("=======================================================")
    print("Number of sample at coarsest grid")
    N0_vec = sensitivity_df["N0"].values
    N0_vec = list(set(N0_vec))
    N0_vec.sort()
    df_N0_list = list()
    for sample_0 in N0_vec:
        index_list_i = sensitivity_df.index[sensitivity_df.N0 == sample_0]
        new_df = sensitivity_df.ix[index_list_i]
        df_N0_list.append(new_df)

    print("=======================================================")
    print("Model Complexity")
    comp_vec = sensitivity_df["Model Complexity"].values
    comp_vec = list(set(comp_vec))
    comp_vec.sort()
    df_comp_list = list()
    for comp in comp_vec:
        index_list_i = sensitivity_df.index[sensitivity_df["Model Complexity"] == comp]
        new_df = sensitivity_df.ix[index_list_i]
        df_comp_list.append(new_df)

        df_for_time = new_df
        mod_good_time = list()
        err_list = list()
        df_for_time = df_for_time.loc[df_for_time["MPE"]>max(min(score_fin_list), min(df_for_time["MPE"].values)) ]
        df_for_time = df_for_time.loc[df_for_time["MPE"] < min(max(score_fin_list), max(df_for_time["MPE"].values))]
        print(df_for_time)
        for i in range(len(df_for_time)):
            value_MPE_multi = df_for_time["MPE"].values[i]
            time_multi = df_for_time["Time"].values[i]
            value_MPE_single = min(score_fin_list, key=lambda x: abs(x - value_MPE_multi))
            idx = score_fin_list.index(min(score_fin_list, key=lambda x: abs(x - value_MPE_multi)))
            time_single = time_fin_list[idx]
            ratio_time = time_single[0]/time_multi
            adj_mpe = value_MPE_multi / value_MPE_single
            ratio_time_adj = ratio_time / adj_mpe
            mod_good_time.append(ratio_time_adj)
            err_list.append(value_MPE_multi)
        df_for_time["err"] = err_list
        df_for_time["rat"] =mod_good_time


        plt.figure()
        plt.scatter(err_list, mod_good_time)



    out_var_vec = list()
    out_var_vec.append("Model Goodness MPE")
    # out_var_vec.append("MPE")
    # out_var_vec.append("SPE")
    # out_var_vec.append("gain_MPE_coar")
    # out_var_vec.append("gain_SPE_coar")
    # out_var_vec.append("gain_MPE_fin")
    # out_var_vec.append("gain_SPE_fin")
    # out_var_vec.append("Time")
    # out_var_vec.append("Efficiency")
    # out_var_vec.append("Model Goodness SPE")


    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    ax.set_axisbelow(True)
    ax = sns.distplot(sensitivity_df["Model Goodness MPE"], kde=True, hist=True, norm_hist=False,kde_kws = {'shade': True, 'linewidth': 2})
    baseline = 1
    plt.gca().set_xlim(left=0)
    plt.axvspan(-1, baseline, alpha=0.25, color='grey')
    # Annotate
    x_line_annotation = baseline
    x_text_annotation = baseline
    plt.text(x =0.1, y=0.1,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes,
             s='Baseline\n' + r'$G = 1$',
             rotation=0,
             bbox=dict(boxstyle="round", ec=(0, 0, 0), fc=(0.95, 0.95, 0.95),))
    plt.xlabel(r'Accuracy Speed Up $G$')

    #total_list = [df_N0_list, df_Nf_list,df_comp_list, df_MPE_list]
    #name_list = [r'$M_0$', r'$M_L$', r'$c_{ml}$', r'$\leq \varepsilon_G <$']
    #var_list = ["N0", "Nf", "Model Complexity", "null"]

    total_list = [df_N0_list, df_Nf_list,df_comp_list]
    name_list = [r'$N_0$', r'$N_L$', r'$c_{ml}$']
    var_list = ["N0", "Nf", "Model Complexity"]

    print(out_var_vec)

    if variable_name == "x_max":
        remove_outliers = 22
    elif variable_name == "Lift":
        remove_outliers = 3
    elif variable_name == "Drag":
        remove_outliers = 6
    for our_var in out_var_vec:
        print(our_var)
        for j in range(len(var_list)):
            var = var_list[j]
            name = name_list[j]
            sens_list = total_list[j]
            Nf_dep_fig = plt.figure()
            axes = plt.gca()
            max_val = 0
            plt.grid(True, which="both", ls=":")
            for i in range(len(sens_list)):
                df = sens_list[i]

                if j != 3:
                    value = df[var].values[0]
                    label = name + r' $=$ ' + str(value)
                    print("#################################")

                    print(var, df[var].values[0])
                    print(df[our_var].loc[(df["Model Goodness MPE"] < remove_outliers)].mean())
                #else:
                #    value_min = min_MPE_list[i]
                #    value_max = max_MPE_list[i]
                #    label = str(value_min) + r' $\leq \varepsilon_G <$ ' + str(value_max)

                if j != 3:
                    sns.distplot(df[our_var], label=label, kde=True, hist=False, norm_hist=False,kde_kws={'shade': True, 'linewidth': 2})
                #else:
                    #sns.distplot(df["Model Goodness MPE"], label=label, kde=True, hist=False, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
                    #plt.gca().set_ylim(bottom=0)
            plt.legend(loc=1)

            if "gain" in our_var or "Good" in our_var or "Base" in our_var:

                baseline = 1
                plt.gca().set_xlim(left=0)
                if  variable_name=="Drag":
                    plt.gca().set_xlim(right=8)
                plt.axvspan(-1, baseline, alpha=0.25, color='grey')
                # Annotate
                x_line_annotation = baseline
                x_text_annotation = baseline
                plt.text(x =0.1, y=0.1,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes.transAxes,
                         s='Baseline\n' + r'$G = 1$',
                         rotation=0,
                         #weight="bold",
                         bbox=dict(boxstyle="round", ec=(0, 0, 0), fc=(0.95, 0.95, 0.95),))
                plt.xlabel(r'Accuracy Speed Up $G$')

            #plt.savefig('C:\\Users\\rober\\Desktop\\LA Presentation\\sens_' + variable_name + "_" + str(k)+"_" + str(j) + '.png', dpi=500)

    if keyword == "airf":
        perc = 4
    elif keyword == "parab":
        perc = 10
    print("Percentage good:",len(sensitivity_df.loc[(sensitivity_df["Model Goodness MPE"]>1)])/len(sensitivity_df["Model Goodness MPE"])*100)
    print("Percentage larger than "+str(perc)+": "+ str(len(sensitivity_df.loc[(sensitivity_df["Model Goodness MPE"]>perc)])/len(sensitivity_df["Model Goodness MPE"])*100)+"%")


    ##########################################################################################################

    fig_scatt_MPE = plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    n_roll = 14
    plt.scatter(sensitivity_df["Time"], sensitivity_df["MPE"],marker="v", color="DarkRed", label="Multi Level Model")
    plt.scatter(time_fin_list, score_fin_list, marker="v", color="DarkBlue", label="Single Level Model")
    plt.legend(loc=3)
    plt.xlabel("Computational Time")
    plt.ylabel("Mean Prediction Error")
    plt.yscale('log')
    plt.xscale('log')
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\mpe_time_tot_' + variable_name  + '.pdf', format='pdf')

    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    palette = sns.color_palette("coolwarm", len(df_Nf_list))
    for i in range(len(df_Nf_list)):
        plt.scatter(df_Nf_list[i]["Time"], df_Nf_list[i]["MPE"], label=r'$N_L = $ ' +str(Nf_vec[i]))
    plt.scatter(time_fin_list, score_fin_list, marker="v", color="DarkBlue", label="Single Level Model")
    plt.legend(loc=3)
    plt.xlabel("Computational Time")
    plt.ylabel("Mean Prediction Error")
    plt.yscale('log')
    plt.xscale('log')
    x_value=[str(x)+ r'\%' for x in ax.get_yticks()]
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\mpe_time_' + variable_name + "_" + str(0) + '.pdf', format='pdf')

    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    for i in range(len(df_N0_list)):
        #plt.scatter(np.log10(sensitivity_df["Time"]), np.log10(sensitivity_df["MPE"]), label="Multi Level Model", alpha=0.5)
        plt.scatter(df_N0_list[i]["Time"], df_N0_list[i]["MPE"], label=r'$N_0 = $ '+str(N0_vec[i]))
    plt.scatter((time_fin_list), (score_fin_list), marker="v", color="DarkBlue", label="Single Level Model")
    plt.legend(loc=3)
    plt.xlabel("Computational Time")
    plt.ylabel("Mean Prediction Error")
    plt.yscale('log')
    plt.xscale('log')
    x_value=[str(x)+ r'\%' for x in ax.get_yticks()]
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\mpe_time_' + variable_name + "_" + str(1) + '.pdf', format='pdf')

    plt.figure()
    ax = plt.gca()
    plt.grid(True, which="both", ls=":")
    for i in range(len(df_comp_list)):
        #plt.scatter(np.log10(sensitivity_df["Time"]), np.log10(sensitivity_df["MPE"]), label="Multi Level Model", alpha=0.5)
        plt.scatter((df_comp_list[i]["Time"]), (df_comp_list[i]["MPE"]), label=r'$c_{ml} = $ '+str(comp_vec[i]))
    plt.scatter((time_fin_list), (score_fin_list), marker="v", color="DarkBlue", label="Single Level Model")
    plt.legend(loc=3)
    plt.xlabel("Computational Time")
    plt.ylabel("Mean Prediction Error")
    plt.yscale('log')
    plt.xscale('log')
    x_value=[str(x)+ r'\%' for x in ax.get_yticks()]
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.savefig('C:\\Users\\rober\\Desktop\\Last Semester\\Master Thesis\\Project\\LA Presentation\\mpe_time_' + variable_name + "_" + str(2) + '.pdf', format='pdf')

    plt.show()
