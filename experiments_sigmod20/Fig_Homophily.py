import numpy as np
import datetime
import random
import os                       # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   to_centering_beliefs,
                   eps_convergence_linbp_parameterized,
                   matrix_difference,
                   matrix_difference_classwise,
                   introduce_errors,
                   showfig)
from estimation import (estimateH,
                             estimateH_baseline_serial,
                             estimateH_baseline_parallel)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from graphGenerator import planted_distribution_model_H
from inference import linBP_symmetric_parameterized, beliefPropagation

# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')

def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False):
    CHOICE = choice

    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    SHOW_PDF = show_pdf
    CREATE_PDF = create_pdf

    STD_FILL = True
    #
    SHORTEN_LENGTH = False

    fig_filename = 'Fig_homophily_{}.pdf'.format(CHOICE)
    csv_filename = 'Fig_homophily_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',
              'f',
              'accuracy']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # -- Default Graph parameters
    k = 3
    rep_DifferentGraphs = 1
    rep_SameGraph = 2
    initial_h0 = None
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    constraint = True

    variant = 1
    EC = True                   # Non-backtracking for learning
    global f_vec, labels, facecolor_vec

    s = 0.5
    err = 0
    numMaxIt = 10
    avoidNeighbors = False
    convergencePercentage_W = None
    stratified = True


    clip_on_vec = [True] * 10
    draw_std_vec = range(10)
    ymin = 0.3
    ymax = 1
    xmin = 0.001
    xmax = 1
    xtick_lab = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    xtick_labels = ['1e-5', '0.01\%', '0.1\%', '1\%', '10\%', '100\%']
    ytick_lab = np.arange(0, 1.1, 0.1)
    linestyle_vec = ['dashed'] + ['solid'] * 10
    linewidth_vec = [5, 2, 3, 3, 3, 3] + [3]*10
    marker_vec = [None, '^', 'v', 'o', '^'] + [None]*10
    markersize_vec = [0, 8, 8, 8, 6, 6] + [6]*10
    facecolor_vec = ['black', "#C44E52",  "#64B5CD"]


    # -- Options with propagation variants
    if CHOICE == 101:
        n = 10000
        h = 3
        d = 15
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GT','DHE','Homophily']
        weight_vec = [None] + [10] + [None]
        randomize_vec = [None] + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GS', 'DCEr', 'Homophily']

    else:
        raise Warning("Incorrect choice!")

    a = 1
    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)

    H0 = create_parameterized_H(k, h, symmetric=True)
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))


    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for i in range(rep_DifferentGraphs):  # create several graphs with same parameters

            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d, distribution=distribution,
                                                      exponent=exponent, directed=False, debug=False)
            X0 = from_dictionary_beliefs(Xd)

            for j in range(rep_SameGraph):  # repeat several times for same graph
                # print("Graph:{} and j: {}".format(i,j))

                ind = None
                for f in f_vec:
                    X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, ind_prior=ind, stratified=stratified)
                    X2 = introduce_errors(X1, ind, err)

                    for option_index, (option, learning_method,  weights, randomize) in \
                            enumerate(zip(option_vec, learning_method_vec, weight_vec, randomize_vec)):

                        # -- Learning
                        if learning_method == 'GT':
                            H2 = H0
                        elif learning_method == 'Homophily':
                            H2 = np.identity(k)

                        elif learning_method == 'DHE':
                            H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize, constraints=constraint)
                            # print("learning_method:", learning_method)
                            # print("H:\n{}".format(H2))

                        # -- Propagation
                        H2c = to_centering_beliefs(H2)
                        X2c = to_centering_beliefs(X2, ignoreZeroRows=True)

                        try:
                            eps_max = eps_convergence_linbp_parameterized(H2c, W,
                                                                          method='noecho',
                                                                          X=X2)
                            eps = s * eps_max

                            F, actualIt, actualPercentageConverged = \
                                linBP_symmetric_parameterized(X2, W, H2c * eps,
                                                              method='noecho',
                                                              numMaxIt=numMaxIt,
                                                              convergencePercentage=convergencePercentage_W,
                                                              debug=2)
                        except ValueError as e:
                            print (
                            "ERROR: {} with {}: d={}, h={}".format(e, learning_method, d, h))

                        else:
                            accuracy_X = matrix_difference_classwise(X0, F, ignore_rows=ind)


                            tuple = [str(datetime.datetime.now())]
                            text = [option_vec[option_index],
                                    f,
                                    accuracy_X]
                            tuple.extend(text)
                            # print("option: {}, f: {}, actualIt: {}, accuracy: {}".format(option_vec[option_index], f, actualIt, accuracy_X))
                            save_csv_record(join(data_directory, csv_filename), tuple)



    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))
    desred_decimals = 7
    df1['f'] = df1['f'].apply(lambda x: round(x,desred_decimals))                   # rounding due to different starting points
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))


    # Aggregate repetitions
    df2 = df1.groupby(['option', 'f']).agg \
        ({'accuracy': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(10)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['f'], columns=['option'], values=['accuracy_mean', 'accuracy_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(10)))

    # Extract values
    X_f = df3['f'].values                     # plot x values
    Y=[]
    Y_std=[]
    for option in option_vec:
        Y.append(df3['accuracy_mean_{}'.format(option)].values)
        if STD_FILL:
            Y_std.append(df3['accuracy_std_{}'.format(option)].values)


    if SHORTEN_LENGTH:
        SHORT_FACTOR = 2        ## KEEP EVERY Nth ELEMENT
        X_f  = np.copy(X_f[list(range(0, len(X_f), SHORT_FACTOR)), ])

        for i in range(len(Y)):
            Y[i] = np.copy(Y[i][list(range(0, len(Y[i]), SHORT_FACTOR)), ])
            if STD_FILL:
                Y_std[i] = np.copy(Y_std[i][list(range(0, len(Y_std[i]), SHORT_FACTOR)),])






    if CREATE_PDF or SHOW_PLOT or SHOW_PDF:

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
            for choice, (option, facecolor) in enumerate(zip(option_vec, facecolor_vec)):
                ax.fill_between(X_f, Y[choice] + Y_std[choice], Y[choice] - Y_std[choice],
                                facecolor=facecolor, alpha=0.2, edgecolor=None, linewidth=0)
                ax.plot(X_f, Y[choice] + Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')
                ax.plot(X_f, Y[choice] - Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')

        for choice, (option, label, color, linewidth, clip_on, linestyle, marker, markersize) in \
                enumerate(zip(option_vec, labels, facecolor_vec, linewidth_vec, clip_on_vec, linestyle_vec, marker_vec, markersize_vec)):
            P = ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                    markersize=markersize, markeredgewidth=1, clip_on=clip_on, markeredgecolor='black')

        plt.xscale('log')

        # -- Title and legend
        distribution_label = '$'
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        n_label = '{}k'.format(int(n / 1000))
        if n < 1000:
            n_label='{}'.format(n)
        a_label = ''
        if a != 1:
            a_label = ', a\!=\!{}'.format(a)

        titleString = r'$\!\!\!n\!=\!{}, d\!=\!{}, h\!=\!{}{}{}'.format(n_label, d, h, a_label, distribution_label)
        plt.title(titleString)

        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles, labels,
                            loc='upper left',     # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.3,  # padding inside legend box
                            numpoints=1,  # put the marker only once
                            )
        # # legend.set_zorder(1)
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8

        plt.xticks(xtick_lab, xtick_labels)
        plt.yticks(ytick_lab, ytick_lab)


        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        xlabel(r'Label Sparsity $(f)$', labelpad=0)      # labelpad=0
        ylabel(r'Accuracy', labelpad=0)

        xlim(xmin, xmax)
        ylim(ymin, ymax)

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
    run(101, show_plot=True)
