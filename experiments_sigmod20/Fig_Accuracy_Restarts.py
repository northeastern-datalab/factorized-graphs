
import os
import sys
sys.path.append('../sslh')
sys.path.append('../')
import numpy as np
import random
import datetime
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   matrix_difference_classwise,
                   eps_convergence_linbp_parameterized,
                   to_centering_beliefs,
                   showfig)
from estimation import estimateH, transform_HToh, transform_hToH
from graphGenerator import planted_distribution_model_H
from inference import linBP_symmetric_parameterized
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'

from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False):

    verbose = False
    repeat_diffGraph = 1000
    SUBSET = True
    NOGT = False        ## Not draw Ground Truth Comparison
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    SHOW_PDF = show_pdf
    CREATE_PDF = create_pdf

    STD_FILL = False

    csv_filename = 'Fig_fast_optimal_restarts_Accv2_{}.csv'.format(CHOICE)
    fig_filename = 'Fig_fast_optimal_restarts_Accv2_{}.pdf'.format(CHOICE)
    header = ['currenttime',
              'k',
              'restarts',
              'accuracy']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)




    # -- Default Graph parameters
    global f_vec, labels, facecolor_vec
    global number_of_restarts



    initial_h0 = None
    distribution = 'powerlaw'
    exponent = -0.3  # for powerlaw
    length = 4  # path length
    constraint = True
    gradient = True
    variant = 1
    EC = True
    delta = 0.001
    numMaxIt = 10
    avoidNeighbors = False
    convergencePercentage_W = None
    stratified = True

    learning_method = 'DHE'
    weights = 10
    randomize = True
    return_min_energy = True
    number_of_restarts = [8, 6, 5, 4]



    clip_on_vec = [True] * 20
    draw_std_vec = range(10)
    ymin = 0.3
    ymax = 1
    xmin = 0.001
    xmax = 1
    xtick_lab = []
    xtick_labels = []
    ytick_lab = np.arange(0, 1.1, 0.1)
    linestyle_vec = ['solid','solid','solid'] * 20
    linewidth_vec = [4,4,4,4]*10
    marker_vec = ['x', 'v', '^', '+', '>', '<'] *10
    markersize_vec = [10, 8, 8, 8 ,8 ,8 ,8 ]*10
    facecolor_vec = ["#C44E52", "#4C72B0", "#8172B2",  "#CCB974",  "#55A868", "#64B5CD"]*5




    # -- Options mainly change k

    if CHOICE == 101:
        n = 10000
        h = 3
        d = 15
        k_vec = [3, 4, 5, 6, 7, 10, 13, 16, 18, 20]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [30, 20, 10, 7, 5, 4, 3, 2, 1, 50, 99, 100]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]


    elif CHOICE == 102:
        n = 10000
        h = 3
        d = 15
        k_vec = [3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        # number_of_restarts = [30, 20, 10, 7, 5, 4, 3, 2, 1, 50, 99, 100]

        number_of_restarts = [20, 10, 5, 4, 3, 2]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]


    elif CHOICE == 103:
        n = 10000
        h = 3
        d = 15
        k_vec = [3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [20, 10, 5, 4, 3, 2, 99]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['o', 'x', 'v', '^', '+', 's', None] * 10
        markersize_vec = [6, 10, 6, 6, 10, 6] * 10

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]


    elif CHOICE == 104:
        n = 10000
        h = 8
        d = 15
        k_vec = [3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [20, 10, 5, 4, 3, 2, 99]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['o', 'x', 'v', '^', '+', 's', None] * 10
        markersize_vec = [6, 10, 6, 6, 10, 6] * 10

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]



    elif CHOICE == 105:
        n = 10000
        h = 8
        d = 15
        k_vec = [3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [20, 10, 5, 4, 3, 2, 100]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['o', 'x', 'v', '^', '+', 's', None] * 10
        markersize_vec = [6, 10, 6, 6, 10, 6] * 10

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]

    elif CHOICE == 106:
        n = 10000
        h = 3
        d = 15
        k_vec = [3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [20, 10, 5, 4, 3, 2, 100]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['o', 'x', 'v', '^', '+', 's', None] * 10
        markersize_vec = [6, 10, 6, 6, 10, 6] * 10

        labels = ['r' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]


    elif CHOICE == 107:

        n = 10000
        h = 8
        d = 15
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [10, 5, 4, 3, 2, 99]
        # number_of_restarts = [20, 10, 5, 4, 3, 2, 100]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['x', 'v', '^', 's', 'o',  's', None] * 10
        markersize_vec = [10, 6, 6, 6, 6, 6, 6] * 10

        labels = [r'$r=$' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]

    elif CHOICE == 108:

        n = 10000
        h = 8
        d = 15
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [4, 5, 7, 10]
        f = 0.09
        distribution = 'uniform'

        # Write in DESCENDING ORDER
        number_of_restarts = [10, 5, 4, 3, 2, 99]
        # number_of_restarts = [20, 10, 5, 4, 3, 2, 100]
        ### 100:GT 99:GTr
        ### 50:min{30,GTr} 1:uninformative

        marker_vec = ['x', 'v', '^', 's', 'o',  's', None] * 10
        markersize_vec = [10, 6, 6, 6, 6, 6, 6] * 10

        labels = [r'$r=$' + str(a1) for a1 in number_of_restarts]
        xtick_lab = k_vec
        xtick_labels = [str(a1) for a1 in k_vec]
        repeat_diffGraph = 10

    else:
        raise Warning("Incorrect choice!")

    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))



    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for _ in range(repeat_diffGraph):

            for k in k_vec:
                a = [1.] * k
                k_star = int(k * (k - 1) / 2)
                alpha0 = np.array(a)
                alpha0 = alpha0 / np.sum(alpha0)

                # Generate Graph
                # print("Generating Graph: n={} h={} d={} k={}".format(n, h, d, k))
                H0 = create_parameterized_H(k, h, symmetric=True)
                W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d, distribution=distribution, exponent=exponent, directed=False, debug=False)
                H0_vec = transform_HToh(H0)
                # print("\nGold standard {}".format(np.round(H0_vec, decimals=3)))

                X0 = from_dictionary_beliefs(Xd)
                X2, ind = replace_fraction_of_rows(X0, 1 - f, avoidNeighbors=avoidNeighbors, W=W, ind_prior=None, stratified=stratified)

                h0 = [1.] * int(k_star)
                h0 = np.array(h0)
                h0 = h0 / k

                delta = 1 / (3 * k)
                # print("delta: ", delta)

                perm = []
                while len(perm) < number_of_restarts[0]:
                    temp = []
                    for _ in range(k_star):
                        temp.append(random.choice([-delta, delta]))
                    if temp not in perm:
                        perm.append(temp)
                    if len(perm) >= 2 ** (k_star):
                        break

                E_list = []   ## format = [[energy, H_vec], []..]
                for vec in perm:
                    H2_vec, energy = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC,
                                               weights=weights, randomize=False, constraints=constraint,
                                               gradient=gradient, return_min_energy=True, verbose=verbose,
                                               initial_h0=h0 + np.array(vec))
                    E_list.append([energy, list(H2_vec)])

                # print("All Optimizaed vector:")
                # [print(i) for i in E_list ]

                # print("Outside Energy:{} optimized vec:{} \n".format(min_energy_vec[0], optimized_Hvec))

                # min_energy_vec = min(E_list)
                # optimized_Hvec = min_energy_vec[1]
                #
                # print("\nEnergy:{} optimized vec:{}  \n\n".format(min_energy_vec[0],optimized_Hvec))
                #
                #

                GTr_optimized_Hvec, GTr_energy = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC,
                                                   weights=weights, randomize=False, constraints=constraint,
                                                   gradient=gradient, return_min_energy=True, verbose=verbose,
                                                   initial_h0=H0_vec)

                uninformative_optimized_Hvec, uninformative_energy = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC,
                                                   weights=weights, randomize=False, constraints=constraint,
                                                   gradient=gradient, return_min_energy=True, verbose=verbose,
                                                   initial_h0=h0)


                iterative_permutations = list(E_list)
                for restartz in number_of_restarts:
                    if k==2 or k == 3 and restartz > 8 and restartz<99:
                        continue

                    if restartz <= number_of_restarts[0]:
                        iterative_permutations = random.sample(iterative_permutations, restartz)
                    # print("For restart:{}, we have vectors:\n".format(restartz))
                    # [print(i) for i in  iterative_permutations]


                    if restartz == 100:       ## for GT
                        H2c = to_centering_beliefs(H0)
                        # print("\nGT: ", transform_HToh(H0,k))

                    elif restartz == 99:       ## for DCEr init with GT
                        H2c = to_centering_beliefs(transform_hToH(GTr_optimized_Hvec, k))
                        # print("\nGTr: ", GTr_optimized_Hvec)

                    elif restartz == 1:  ## for DCEr with uninformative initial
                        H2c = to_centering_beliefs(transform_hToH(uninformative_optimized_Hvec, k))
                        # print("\nUninformative: ", uninformative_optimized_Hvec)

                    elif restartz == 50:  ## for min{DCEr , GTr}
                        # print("Length:",len(E_list))
                        # [print(i) for i in E_list]
                        mod_E_list = list(E_list)+[[GTr_energy , list(GTr_optimized_Hvec)]]     #Add GTr to list and take min
                        # print("Mod Length:", len(mod_E_list))
                        # [print(i) for i in mod_E_list]
                        min_energy_vec = min(mod_E_list)
                        # print("\nSelected for 50:",min_energy_vec)
                        optimized_Hvec = min_energy_vec[1]

                        H2c = to_centering_beliefs(transform_hToH(optimized_Hvec, k))

                    else:
                        min_energy_vec = min(iterative_permutations)
                        optimized_Hvec = min_energy_vec[1]
                        H2c = to_centering_beliefs(transform_hToH(optimized_Hvec, k))

                    # print("Inside Chosen Energy:{} optimized vec:{} \n".format(min_energy_vec[0], optimized_Hvec))

                    try:
                        eps_max = eps_convergence_linbp_parameterized(H2c, W, method='noecho', X=X2)
                        s = 0.5
                        eps = s * eps_max

                        F, actualIt, actualPercentageConverged = \
                            linBP_symmetric_parameterized(X2, W, H2c * eps,
                                                          method='noecho',
                                                          numMaxIt=numMaxIt,
                                                          convergencePercentage=convergencePercentage_W,
                                                          debug=2)
                    except ValueError as e:
                        print(
                            "ERROR: {} with {}: d={}, h={}".format(e, learning_method, d, h))

                    else:
                        acc = matrix_difference_classwise(X0, F, ignore_rows=ind)

                        tuple = [str(datetime.datetime.now())]
                        text = [k,
                                restartz,
                                acc]
                        tuple.extend(text)

                        if verbose:
                            print("\nGold standard {}".format(np.round(H0_vec, decimals=3)))
                        # print("k:{}  Restart:{}  OptimizedVec:{}  Energy:{}  Accuracy:{}".format(k, restartz, np.round(min_energy_vec[1], decimals=3), min_energy_vec[0], acc  ))
                        # print("k:{}  Restart:{}   Accuracy:{}".format(k, 1, L2_dist))
                        save_csv_record(join(data_directory, csv_filename), tuple)



    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1 (length {}):\n{}".format(len(df1.index), df1.head(20)))

    # Aggregate repetitions
    df2 = df1.groupby(['k', 'restarts']).agg \
        ({'accuracy': [np.mean, np.std, np.size], })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    df2['restarts'] = df2['restarts'].astype(str)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(20)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['k'], columns=['restarts'], values=['accuracy_mean', 'accuracy_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(10)))




    df4 = df3.drop('k', axis=1)
    if NOGT:
        df4 = df3.drop(['k', 'accuracy_mean_0', 'accuracy_mean_1', 'accuracy_std_0', 'accuracy_std_1'], axis=1)

    # df4 = df3.drop(['k', 'accuracy_mean_100', 'accuracy_std_100'], axis=1)


    df5 = df4.div(df4.max(axis=1), axis=0)
    df5['k'] = df3['k']
    # print("\n-- df5 (length {}):\n{}".format(len(df5.index), df5.head(100)))

    # df5 = df3     ## for normalization

    X_f = df5['k'].values            # read k from values instead
    Y=[]
    Y_std=[]
    for rez in number_of_restarts:
        if NOGT:
            if rez == 100 or rez==99:
                continue
        Y.append(df5['accuracy_mean_{}'.format(rez)].values)
        if STD_FILL:
            Y_std.append(df5['accuracy_std_{}'.format(rez)].values)



    if CREATE_PDF or SHOW_PDF or SHOW_PLOT:

        # -- Setup figure
        mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 14
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
        mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['figure.figsize'] = [4, 4]
        fig = figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])


        #  -- Drawing
        if STD_FILL:
            for choice, (option, facecolor) in enumerate(zip(number_of_restarts, facecolor_vec)):
                if option == 100:  ## GT
                    if NOGT:
                        continue
                    facecolor = 'black'
                elif option == 99:  ## GT-r
                    if NOGT:
                        continue
                    facecolor = 'black'

                ax.fill_between(X_f, Y[choice] + Y_std[choice], Y[choice] - Y_std[choice],
                                facecolor=facecolor, alpha=0.2, edgecolor=None, linewidth=0)
                ax.plot(X_f, Y[choice] + Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')
                ax.plot(X_f, Y[choice] - Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')

        for choice, (option, label, color, linewidth, clip_on, linestyle, marker, markersize) in \
                enumerate(zip(number_of_restarts, labels, facecolor_vec, linewidth_vec, clip_on_vec, linestyle_vec, marker_vec, markersize_vec)):

            if option == 100:     ## GT
                if NOGT:
                    continue
                linestyle='dashed'
                linewidth=3
                color='black'
                label='GS'
                marker='x'
                markersize=6
            elif option == 99:       ## GT-r
                if NOGT:
                    continue
                linestyle='dashed'
                linewidth=2
                color='black'
                label='Global Minima'
                marker = None
                markersize = 6
            elif option == 1:     ## GT
                color="#CCB974"
                linewidth = 2
                label='Uninfo'
            elif option == 50:       ## GT-r
                label='min{30,GTr}'

            P = ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                    markersize=markersize, markeredgecolor='black',  markeredgewidth=1, clip_on=clip_on)

        # plt.xscale('log')

        # -- Title and legend
        distribution_label = '$'
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        n_label = '{}k'.format(int(n / 1000))
        if n < 1000:
            n_label='{}'.format(n)

        titleString = r'$\!\!\!n\!=\!{}, d\!=\!{}, h\!=\!{}, f\!=\!{} $'.format(n_label, d, h, f)
        title(titleString)

        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles, labels,
                            loc='lower left',     # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.3,  # padding inside legend box
                            numpoints=1,  # put the marker only once
                            # bbox_to_anchor=(1.1, 0)
                            )
        # # legend.set_zorder(1)
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8

        plt.xticks(xtick_lab, xtick_labels)
        # plt.yticks(ytick_lab, ytick_lab)


        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        # ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))

        grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        xlabel(r'Number of Classes $(k)$', labelpad=0)      # labelpad=0
        ylabel(r'Relative Accuracy', labelpad=0)

        xlim(2.9, 7.1)
        #
        ylim(0.65, 1.015)

        if CREATE_PDF:
            savefig(join(figure_directory, fig_filename), format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05,
                    frameon=None)
        if SHOW_PLOT:
            plt.show()

        if SHOW_PDF:
            showfig(join(figure_directory, fig_filename))  # shows actually created PDF



if __name__ == "__main__":
    run(107, show_plot=True)
