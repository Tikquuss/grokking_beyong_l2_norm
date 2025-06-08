import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

#from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

FIGSIZE = (6, 4)
FIGSIZE_SMALL = (6, 4)
FIGSIZE_MEDIUM = (8, 6)
FIGSIZE_LARGE = (15, 10)

LINEWIDTH = 2.0
FONTSIZE = 12

LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 15

MARKERSIZE = 10

def get_training_phases(all_steps, train_accs, test_accs, min_acc=0.05, max_acc=0.99):
    t_1, t_2, t_3, t_4 = None, None, None, None
    if train_accs is not None :
        for t, acc in zip(all_steps, train_accs):
            if acc>min_acc : t_1 = t; break
        for t, acc in zip(all_steps, train_accs):
            if acc>=max_acc : t_2 = t; break
    if test_accs is not None :
        for t, acc in zip(all_steps, test_accs):
            if acc>min_acc : t_3 = t; break
        for t, acc in zip(all_steps, test_accs):
            if acc>=max_acc : t_4 = t; break
    return t_1, t_2, t_3, t_4

def find_closest_step(step, all_steps) :
    if step is None: return None
    all_steps_step_index = {k:v for v, k in enumerate(all_steps)}
    candidates = np.array(list(all_steps_step_index.keys()))
    # Find the closest checkpoint step with acc compute
    closest = candidates[np.abs(candidates - step).argmin()]
    index = all_steps_step_index[closest]
    closest_step = all_steps[index]
    # train_acc = train_accs[index]
    # val_acc = test_accs[index]
    return closest_step

###################### .... ###################### 

def get_twin_axis(
    ax=None, color_1="k", color_2="k", no_twin=False,
    axis = "x",
    linewidth=0.8, # 0.3 # major
    linewidth_minor=0.2, # 0.2 # minor
    alpha=0.7,
    alpha_minor=0.3,
    ) :

    assert axis in ["x", "y"], "axis must be 'x' or  'y'"
    axis_second = "y" if axis == "x" else "x"

    color='black' # major : 'k' 'gray' 'black' ...
    color_minor='gray' # minor
    linestyle="-"
    linestyle_minor='--'

    if ax is None :
        R, C = 1, 1
        #figsize=(C*15, R*10)
        figsize=(C*6, R*4)
        fig, ax1 = plt.subplots(figsize=figsize)
    else :
        ax1 = ax
        fig = None

    ax1.grid(axis=axis, linestyle=linestyle, which='major', color=color, linewidth=linewidth, alpha=alpha)
    ax1.grid(axis=axis, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    ax1.grid(axis=axis_second, linestyle=linestyle, which='major', color=color_1, linewidth=linewidth, alpha=alpha)
    ax1.grid(axis=axis_second, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    plt.minorticks_on()

    if no_twin : return fig, ax1, None

    ax2 = ax1.twinx() if axis == "x" else ax1.twiny()    
    ax2.grid(axis=axis_second, linestyle=linestyle, which='major', color=color_2, linewidth=linewidth, alpha=alpha)
    ax2.grid(axis=axis_second, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    plt.minorticks_on()

    return fig, ax1, ax2


def add_legend(element, legend_elements, labels,
              #bbox_to_anchor=(0., 1.02, 1., .102), # top
              #bbox_to_anchor=(0.5, 1.09), # top
              #bbox_to_anchor=(0.5, 1.05), # top, on the line
              loc='best',
               ):

    # legend_elements_train = (Line2D([0], [0], linestyle='-', color=color_train))
    # legend_elements_val = (Line2D([0], [0], linestyle='-', color=color_val))
    # legend_elements = [legend_elements_train, legend_elements_val]
    # labels = [train_label, val_label]

    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    # locations : 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    element.legend(legend_elements, labels,
                    loc=loc,
                    #ncol=2,
                    #bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
                    handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=2,
                    fontsize='large',
                )