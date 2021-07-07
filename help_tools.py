import datetime, numpy as np, pandas as pd
import scipy.stats as st, sklearn
import matplotlib.pyplot as plt
import matplotlib
import traceback
import imp
import string
from hashlib import sha256
import math
from statsmodels.stats.weightstats import ztest, ttest_ind
from scipy import stats
from scipy.stats import ttest_ind as ttest_ind_from_scipy
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power

def init_matplot(figsize_xy=(15,5), subplot_grid=None):
    """Преднастройка отображения графиков в plt"""
    matplotlib.rcParams['figure.figsize'] = figsize_xy
    matplotlib.rcParams['figure.titlesize'] = 15
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['axes.titlesize'] = 15
    matplotlib.rcParams['axes.labelsize'] = 15
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['lines.linewidth'] = 2
    # сетка графиков - на выходе fig, list_ax[n,m]
    # управление через plt.sca(ax)
    if subplot_grid:
        m, n = subplot_grid
        return plt.subplots(m, n)

def mean_ttest(list_1, list_2, significance = 0.05):
    """в пределе T-распределение сходится к Z-распредлению"""
    T, p_value, _ = ttest_ind(list_2, list_1, alternative='larger', usevar='unequal')
    if p_value <= significance / 2:
        decision = 'M(list_2) > M(list_1)'
    elif p_value >= 1 - significance / 2:
        decision = 'M(list_2) < M(list_1)'
    else:
        decision = 'M(list_2) ~ M(list_1)'
    return p_value, np.mean(list_2) - np.mean(list_1), decision
