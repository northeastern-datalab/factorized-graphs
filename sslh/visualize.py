"""
Plotting of various figures
"""

import os                       # for displaying created PDF
import platform
import random

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



plat = platform.system()
pdf_open = {'Darwin': 'open', 'Windows': 'start', 'Linux': 'xdg-open'}


def plot(data, path, n=None, d=None, k=None,
    xmin=None, ymin=None, xmax=None, ymax=None,
    metric='acc', dataset="", fork=True, labels=None,
    line_colors=None, markers=None, line_styles=None, draw_stds=None,
    marker_sizes=None, line_widths=None, legend_location=None, show=True,
    save=False, show_plot=False):
    """ Generates a plot evaluating the performance of sslh learning and
    propagation. Saves the figure to the specified directory and displays it
    using the default system pdf viewer.
    data:      DataFrame with the columns:
               [String,   ( Number,       Number) / ( Number, Number)]
               ['method', ('precision', 'recall') / ('f', 'accuracy')]

    fig_name:  name of the figure, String
    directory: location to save the figure, String
    n:         number of nodes in the graph used to generate the data, Number
    d:         average degree of the graph, Number
    metric:    oneof('acc', 'pr')
    with_std:  True to show standard deviation around metric, Boolean
    colors:    List of colors to use for plotting each method, List of String
    fork:      Will fork the displaying of the pdf into a background process
               [Unix only] Boolean


    """
    x_label = 'f'
    y_label = 'accuracy'
    xtick_lab = [0.001, 0.01, 0.1, 1]
    xtick_labels = ['0.1%', '1%', '10%', '100%']
    ytick_lab = np.arange(0, 1.1, 0.1)
    ytick_labels = ytick_lab

    if ymin is None:    
        if k is None:
            ymin = 0
        else:
            ymin = 1 / k

    if xmin is None:
        xmin = 0.0001
    
    if xmax is None:
        xmax = 1
    
    if ymax is None:
        ymax = 1

    if legend_location is None:
        legend_location = 'lower right'

    if metric == 'pr':
        x_label = 'precision'
        y_label = 'recall'
        xtick_lab = np.arange(0, 1.1, 0.1)
        ytick_lab = np.arange(0, 1.1, 0.1)
        xtick_labels = xtick_lab
        ytick_labels = ytick_lab

        if ymin is None:
            ymin = 0

        if xmin is None:
            xmin = 0
        
        if xmax is None:
            xmax = 1
        
        if ymax is None:
            ymax = 1

        if legend_location is None:
            legend_location = 'lower left'

    # -- Read, aggregate, and pivot data for all options
    # print("\n-- data: (length {}):\n{}".format(len(data.index), data.head(500)))
    if labels == None:
        methods = list(set(data['method'].values))
    else:
        methods = labels

    num_methods = len(methods)
    # print("Methods: {}".format(methods))
    # Aggregate repetitions
    df2 = data.groupby(['method', x_label]).agg({y_label: [np.mean, np.std, np.size],  # Multiple Aggregates
                                                 })
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(500)))
    # flatten the column hierarchy
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={y_label + '_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(500)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=x_label, columns='method', values=[
                         y_label + '_mean', y_label + '_std'], )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(5)))
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    # flatten the column hierarchy
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(5)))

    # Extract values
    X_f = df3[x_label].values                     # plot x values
    Y = []
    Y_std = []
    for method in methods:
        Y.append(df3['{}_mean_{}'.format(y_label, method)].values)
        if draw_stds is not None:
            Y_std.append(df3['{}_std_{}'.format(y_label, method)].values)
    
    # print('Means:\n{}'.format(Y))
    # print('Stds:\n{}'.format(Y_std))
    # -- Setup figure
    mpl.rc('font', **{'family': 'sans-serif',
                      'sans-serif': [u'Arial', u'Liberation Sans']})
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14        # 6
    mpl.rcParams['grid.color'] = '777777'  # grid color
    mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
    mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['figure.figsize'] = [4, 4]
    fig = plt.figure()
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

    # Plotting Parameters
    clip_on_vec = [True] * num_methods
    numMaxIt_vec = [10] * num_methods
    if line_styles is None:
        line_styles = ['solid'] * num_methods
    
    if line_widths is None:
        line_widths = [random.choice([1, 2, 3, 4]) for _ in range(num_methods)]

    if markers is None: 
        markers = [random.choice(['o', 'x', '^', 'v', '+']) for _ in range(num_methods)]

    if marker_sizes is None:
        marker_sizes = [8] * num_methods

    if line_colors is None:
        line_colors = sns.color_palette('colorblind', len(methods))

    #  -- Drawing
    for choice, (method, color, draw_std) in enumerate(zip(methods,
        line_colors, draw_stds)):
        if draw_std:
            # print('Filling between {} +- {}'.format(Y[choice], Y_std[choice]))

            ax.fill_between(X_f, Y[choice] + Y_std[choice], Y[choice] - Y_std[choice],
                            facecolor=color, alpha=0.2, edgecolor=None, linewidth=0)
            ax.plot(X_f, Y[choice] + Y_std[choice],
                    linewidth=0.5, color='0.8', linestyle='solid')
            ax.plot(X_f, Y[choice] - Y_std[choice],
                    linewidth=0.5, color='0.8', linestyle='solid')

    for choice, (method, color, linewidth, clip_on, linestyle, marker, marker_size) in \
            enumerate(zip(methods, line_colors, line_widths, clip_on_vec,
                line_styles, markers, marker_sizes)):
        ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=method, zorder=4, marker=marker,
                markersize=marker_size, markeredgewidth=1, clip_on=clip_on)

    if metric is 'acc':
        plt.xscale('log')

    # -- Title and legend
    title_label = r'$\!\!\!\!\!\!\!${}'.format(dataset.title())
    if n is not None:
        if n < 1000:
            title_label += r': $n={}$'.format(n)
        else:
            title_label += r': $n={}k$'.format(int(n / 1000))

    if d is not None:
        title_label += r', $d={}$'.format(np.round(d, 1))

    plt.title(title_label)
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels,
                        loc=legend_location,     # 'upper right'
                        handlelength=2,
                        labelspacing=0,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        # title='Variants',
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        )
    # # legend.set_zorder(1)
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_alpha(0.9)  # 0.8

    # -- Figure settings and save
    plt.xticks(xtick_lab, xtick_labels)
    plt.yticks(ytick_lab, ytick_labels)

    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid',
               linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
    plt.grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid',
               linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',

    x_label = r'Label Sparsity $(f)$'
    plt.xlabel(x_label, labelpad=0)      # labelpad=0
    plt.ylabel(y_label.title(), labelpad=0)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    if save:
        plt.savefig(path, format='pdf',
                      dpi=None,
                      edgecolor='w',
                      orientation='portrait',
                      transparent=False,
                      bbox_inches='tight',
                      pad_inches=0.05,
                      frameon=None)
    if show:
        if plat in ['Darwin', 'Linux'] and fork:
            # shows actually created PDF
            os.system("{} '{}' &".format(pdf_open[plat], path))
        else:
            # shows actually created PDF
            os.system("{} '{}'".format(pdf_open[plat], path))

    if show_plot:
        plt.show()

