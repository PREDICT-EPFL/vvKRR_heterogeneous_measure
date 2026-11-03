import numpy as np
import matplotlib.pyplot as plt
from regression.models import KRRNonseparableKernel, vvKRRPartialNonseparableKernel, vvKRR
from regression.tools import Tools
from typing import Callable
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors

np.random.seed(0)
def f_true(X:np.ndarray):
    rng = np.random.default_rng(seed=0)
    alpha = rng.uniform(-1, 1, (5, 2))*10
    x_node = np.linspace(0,1,5).reshape(-1,1)
    theta = 0.25*np.pi
    K_XX = kernel_non_separable(x_node,x_node,theta)
    K_xX = kernel_non_separable(X,x_node,theta)
    Y = K_xX@np.linalg.inv(K_XX)@np.concatenate(alpha)
    Y = Y.reshape(-1,2)
    return Y, theta

def kernel_non_separable(
        X_1: np.ndarray,
        X_2: np.ndarray,
        theta
    ):
        A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        n1 = len(X_1)
        n2 = len(X_2)
        blocks = []

        for i in range(n1):
            blocks.append([])
            for j in range(n2):
                D = np.array([[Tools.Kernel.RBF(X_1[i].reshape(1,-1),X_2[j].reshape(1,-1),length_scale = 0.05).squeeze(), 0],[0,Tools.Kernel.RBF(X_1[i].reshape(1,-1),X_2[j].reshape(1,-1),length_scale = 0.5).squeeze()]])
                blocks[i].append((A@D@A.T))
                
        return np.block(blocks)

def get_true_vvf(f_true:Callable):
    X_train = np.linspace(0,1,5).reshape(-1,1)
    Y_train, theta = f_true(X_train)
    
    vvf = vvKRRPartialNonseparableKernel(X_train,[np.eye(2) for i in range(5)], Y_train, theta, Tools.Kernel.RBF, gamma = 5, delta_ub = 0.1, bound = np.array([[0,1]]), lambda_ = 0.01)
    
    return vvf, theta

def sample_unit_ball_n(N, n_samples,rng):
    z = rng.normal(size=(n_samples, N))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / norms
    
    r = rng.random(n_samples) ** (1.0 / N)
    r = r.reshape(-1, 1)
    
    return r * u

def generate_training_data(vvf, scenario_type, X_train=None, delta_ub=0.1):
    rng = np.random.default_rng(seed=0)

    M_full = np.eye(2)
    M1 = np.array([[1., 0.]])
    M2 = np.array([[0., 1.]])
    Msum = np.array([[1., 1.]])

    # Grid scenario: override X_train with linspace
    if scenario_type == 'grid':
        X_train = np.linspace(0, 1, 20)

    num_sample = len(X_train)

    M_list = []
    Y_train = []
    meas_type = []

    X_scalar_1, Y_scalar_1 = [], []
    X_scalar_2, Y_scalar_2 = [], []
    X_scalar_sum, Y_scalar_sum = [], []

    for i, v in enumerate(X_train):
        x = np.atleast_2d(v)
        fx = vvf.get_mean(x,[np.eye(2)])

        if scenario_type == 'grid':
            Mi = M1 if i % 2 == 0 else M2
        else:
            Mi = M_full
            if i == 6:
                if scenario_type == 'single':
                    Mi = M1
                elif scenario_type == 'sum':
                    Mi = Msum

        M_list.append(Mi)

        mi = Mi.shape[0]
        noise_i = sample_unit_ball_n(mi, 1, rng).squeeze() * delta_ub
        yi = Mi @ fx + noise_i
        Y_train.append(yi)

        if Mi.shape[0] == 2:
            meas_type.append('full')
            X_scalar_1.append(v);   Y_scalar_1.append(yi[0])
            X_scalar_2.append(v);   Y_scalar_2.append(yi[1])
            X_scalar_sum.append(v); Y_scalar_sum.append(yi[0] + yi[1])
        elif np.allclose(Mi, M1):
            meas_type.append('first')
            X_scalar_1.append(v);   Y_scalar_1.append(yi[0])
        elif np.allclose(Mi, M2):
            meas_type.append('second')
            X_scalar_2.append(v);   Y_scalar_2.append(yi[0])
        elif np.allclose(Mi, Msum):
            meas_type.append('sum')
            X_scalar_sum.append(v); Y_scalar_sum.append(yi[0])

    X_train_scalar_1   = np.array(X_scalar_1)
    Y_train_scalar_1   = np.array(Y_scalar_1)
    X_train_scalar_2   = np.array(X_scalar_2)
    Y_train_scalar_2   = np.array(Y_scalar_2)
    X_train_scalar_sum = np.array(X_scalar_sum)
    Y_train_scalar_sum = np.array(Y_scalar_sum)

    # col 0: f1, col 1: f2, col 2: f1+f2 (NaN where not observed)
    Y_train_full = np.full((num_sample, 3), np.nan)
    for i, (Mi, yi) in enumerate(zip(M_list, Y_train)):
        if Mi.shape[0] == 2:
            Y_train_full[i, 0] = yi[0]
            Y_train_full[i, 1] = yi[1]
            Y_train_full[i, 2] = yi[0] + yi[1]
        elif np.allclose(Mi, M1):
            Y_train_full[i, 0] = yi[0]
        elif np.allclose(Mi, M2):
            Y_train_full[i, 1] = yi[0]
        elif np.allclose(Mi, Msum):
            Y_train_full[i, 2] = yi[0]

    return (X_train, Y_train, Y_train_full, M_list, meas_type,
            X_train_scalar_1, Y_train_scalar_1,
            X_train_scalar_2, Y_train_scalar_2,
            X_train_scalar_sum, Y_train_scalar_sum)

def run_scenario(vvf, theta, scenario_type, X_train=None):
    (X_train, Y_train, Y_train_full, M_list, meas_type,
     X_train_scalar_1, Y_train_scalar_1,
     X_train_scalar_2, Y_train_scalar_2,
     X_train_scalar_sum, Y_train_scalar_sum) = generate_training_data(vvf, scenario_type, X_train)

    vvKRR_model = vvKRRPartialNonseparableKernel(X_train.reshape(-1,1), M_list, Y_train, theta,
                                Tools.Kernel.RBF,
                                gamma=vvf.norm, delta_ub=0.1,
                                bound=np.array([[0,1]]), lambda_=0.01)

    scalarKRR1 = KRRNonseparableKernel(X_train_scalar_1.reshape(-1,1), Y_train_scalar_1, [np.cos(theta)**2,np.sin(theta)**2], [0.05,0.5],
                     Tools.Kernel.RBF, gamma=vvf.norm,
                     delta_ub=0.1, bound=np.array([[0,1]]), lambda_=0.01)

    scalarKRR2 = KRRNonseparableKernel(X_train_scalar_2.reshape(-1,1), Y_train_scalar_2, [np.sin(theta)**2,np.cos(theta)**2], [0.05,0.5],
                     Tools.Kernel.RBF, gamma=vvf.norm,
                     delta_ub=0.1, bound=np.array([[0,1]]), lambda_=0.01)

    X_grid = np.linspace(0, 1, 100)
    vvKRR_true     = np.array([vvf.get_mean(np.atleast_2d(i), [np.eye(2)])                      for i in X_grid])
    vvKRR_test     = np.array([vvKRR_model.get_mean(np.atleast_2d(i), [np.eye(2)])              for i in X_grid])
    vvKRR_bound1   = np.array([vvKRR_model.get_bound(np.atleast_2d(i), [np.array([[1,0]])])     for i in X_grid])
    vvKRR_bound2   = np.array([vvKRR_model.get_bound(np.atleast_2d(i), [np.array([[0,1]])])     for i in X_grid])
    vvKRR_boundsum = np.array([vvKRR_model.get_bound(np.atleast_2d(i), [np.array([[1,1]])])     for i in X_grid])

    scalarKRR1_test  = np.array([scalarKRR1.get_mean(np.atleast_2d(i))  for i in X_grid])
    scalarKRR1_bound = np.array([scalarKRR1.get_bound(np.atleast_2d(i)) for i in X_grid])
    scalarKRR2_test  = np.array([scalarKRR2.get_mean(np.atleast_2d(i))  for i in X_grid])
    scalarKRR2_bound = np.array([scalarKRR2.get_bound(np.atleast_2d(i)) for i in X_grid])

    return dict(
        X_grid=X_grid,
        X_train=X_train, Y_train_full=Y_train_full, meas_type=meas_type,
        vvKRR_true=vvKRR_true, vvKRR_test=vvKRR_test,
        vvKRR_bound1=vvKRR_bound1, vvKRR_bound2=vvKRR_bound2,
        vvKRR_boundsum=vvKRR_boundsum,
        scalarKRR1_test=scalarKRR1_test, scalarKRR1_bound=scalarKRR1_bound,
        scalarKRR2_test=scalarKRR2_test, scalarKRR2_bound=scalarKRR2_bound,
    )

def plot_scenario_column(axs_col, data, col_idx):
    color_map = {'full': 'black', 'first': 'tab:red', 'second': 'tab:red', 'sum': 'tab:red'}

    X_grid       = data['X_grid']
    X_train      = data['X_train']
    Y_train_full = data['Y_train_full']
    meas_type    = data['meas_type']
    point_colors = [color_map[t] for t in meas_type]

    vvKRR_true = data['vvKRR_true']
    vvKRR_test = data['vvKRR_test']
    s1_test    = data['scalarKRR1_test'];  s1_bound = data['scalarKRR1_bound']
    s2_test    = data['scalarKRR2_test'];  s2_bound = data['scalarKRR2_bound']

    show_ylabel = (col_idx == 0)
    lw = 3.0;  ms = 30;  fs = 12;  fs_tick = 12

    # ---- f1 ----
    ax = axs_col[0]
    mask1 = np.array([t in ('full', 'first') for t in meas_type])
    ax.scatter(X_train[mask1], Y_train_full[mask1, 0],
               c=[point_colors[i] for i in np.where(mask1)[0]],
               edgecolors='white', linewidths=1, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,0], color='black',      linewidth=lw, label='Ground truth')
    ax.plot(X_grid, s1_test,         color='tab:orange',  linewidth=lw, label='Scalar-valued model')
    ax.plot(X_grid, vvKRR_test[:,0], color='tab:blue',    linewidth=lw, label='Vector-valued model')
    ax.fill_between(X_grid, s1_test - s1_bound, s1_test + s1_bound,
                    color='tab:orange', alpha=0.3)
    ax.fill_between(X_grid, vvKRR_test[:,0] - data['vvKRR_bound1'],
                             vvKRR_test[:,0] + data['vvKRR_bound1'],
                    color='tab:blue', alpha=0.3)
    if show_ylabel:
        ax.set_ylabel('$f_1(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_ylim(-35, 30);  ax.grid(True, linewidth=0.4);  ax.tick_params(labelsize=fs_tick)

    # ---- f2 ----
    ax = axs_col[1]
    mask2 = np.array([t in ('full', 'second', 'sum') for t in meas_type])
    ax.scatter(X_train[mask2], Y_train_full[mask2, 1],
               c=[point_colors[i] for i in np.where(mask2)[0]],
               edgecolors='white', linewidths=1, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,1], color='black',      linewidth=lw)
    ax.plot(X_grid, s2_test,         color='tab:orange',  linewidth=lw)
    ax.plot(X_grid, vvKRR_test[:,1], color='tab:blue',    linewidth=lw)
    ax.fill_between(X_grid, s2_test - s2_bound, s2_test + s2_bound,
                    color='tab:orange', alpha=0.3)
    ax.fill_between(X_grid, vvKRR_test[:,1] - data['vvKRR_bound2'],
                             vvKRR_test[:,1] + data['vvKRR_bound2'],
                    color='tab:blue', alpha=0.3)
    if show_ylabel:
        ax.set_ylabel('$f_2(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_ylim(-40, 40);  ax.grid(True, linewidth=0.4);  ax.tick_params(labelsize=fs_tick)

    # ---- f1 + f2 ----
    ax = axs_col[2]
    mask_sum = np.array([t in ('full', 'sum') for t in meas_type])
    ax.scatter(X_train[mask_sum], Y_train_full[mask_sum, 2],
               c=[point_colors[i] for i in np.where(mask_sum)[0]],
               edgecolors='white', linewidths=1, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,0] + vvKRR_true[:,1], color='black',      linewidth=lw)
    ax.plot(X_grid, s1_test + s2_test,                  color='tab:orange',  linewidth=lw)
    ax.plot(X_grid, vvKRR_test[:,0] + vvKRR_test[:,1],  color='tab:blue',    linewidth=lw)
    ax.fill_between(X_grid,
                    (s1_test + s2_test) - (s1_bound + s2_bound),
                    (s1_test + s2_test) + (s1_bound + s2_bound),
                    color='tab:orange', alpha=0.3)
    ax.fill_between(X_grid,
                    (vvKRR_test[:,0] + vvKRR_test[:,1]) - data['vvKRR_boundsum'],
                    (vvKRR_test[:,0] + vvKRR_test[:,1]) + data['vvKRR_boundsum'],
                    color='tab:blue', alpha=0.3)
    if show_ylabel:
        ax.set_ylabel('$f_1(x){+}f_2(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_xlabel('$x$', fontsize=fs)
    ax.set_ylim(-40, 40);  ax.set_xlim(0, 0.5)
    ax.grid(True, linewidth=0.4);  ax.tick_params(labelsize=fs_tick)


def plot_two_scenarios(data_single, data_sum, titles):
    fig, axs = plt.subplots(3, 2, figsize=(7.16, 6.5),
                             sharex=True, sharey='row',
                             gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    axs[0, 0].set_title(titles[0], fontsize=12)
    axs[0, 1].set_title(titles[1], fontsize=12)

    plot_scenario_column(axs[:, 0], data_single, col_idx=0)
    plot_scenario_column(axs[:, 1], data_sum,    col_idx=1)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, fontsize=12, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.14), borderpad=0.4,
               handlelength=1.5, columnspacing=1.0)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.28)
    return fig, axs

def plot_grid_scenario(data):
    lw = 3.0;  ms = 30;  fs = 12;  fs_tick = 12
    color_map    = {'full': 'black', 'first': 'tab:red', 'second': 'tab:red', 'sum': 'tab:red'}
    meas_type    = data['meas_type']
    X_train_g    = data['X_train']
    Y_train_g    = data['Y_train_full']
    point_colors = [color_map[t] for t in meas_type]
    X_grid       = data['X_grid']
    vvKRR_true   = data['vvKRR_true']
    vvKRR_test   = data['vvKRR_test']
    s1_test      = data['scalarKRR1_test'];  s1_bound = data['scalarKRR1_bound']
    s2_test      = data['scalarKRR2_test'];  s2_bound = data['scalarKRR2_bound']

    fig, axs = plt.subplots(2, 1, figsize=(7.16, 4.0),
                             sharex=True, gridspec_kw={'hspace': 0.1})

    # ---- f1 ----
    mask1 = np.array([t in ('full', 'first') for t in meas_type])
    axs[0].scatter(X_train_g[mask1], Y_train_g[mask1, 0],
                   c=[point_colors[i] for i in np.where(mask1)[0]],
                   edgecolors='white', linewidths=1, s=ms, zorder=10)
    axs[0].plot(X_grid, vvKRR_true[:,0], color='black',      linewidth=lw, label='Ground truth')
    axs[0].plot(X_grid, s1_test,         color='tab:orange',  linewidth=lw, label='Scalar-valued model')
    axs[0].plot(X_grid, vvKRR_test[:,0], color='tab:blue',    linewidth=lw, label='Vector-valued model')
    axs[0].fill_between(X_grid, s1_test - s1_bound, s1_test + s1_bound,
                        color='tab:orange', alpha=0.3)
    axs[0].fill_between(X_grid, vvKRR_test[:,0] - data['vvKRR_bound1'],
                                 vvKRR_test[:,0] + data['vvKRR_bound1'],
                        color='tab:blue', alpha=0.3)
    axs[0].set_ylabel('$f_1(x)$', fontsize=fs)
    axs[0].yaxis.set_label_coords(-0.08, 0.5)
    axs[0].set_ylim(-40, 40);  axs[0].grid(True, linewidth=0.4);  axs[0].tick_params(labelsize=fs_tick)

    # ---- f2 ----
    mask2 = np.array([t in ('full', 'second') for t in meas_type])
    axs[1].scatter(X_train_g[mask2], Y_train_g[mask2, 1],
                   c=[point_colors[i] for i in np.where(mask2)[0]],
                   edgecolors='white', linewidths=1, s=ms, zorder=10)
    axs[1].plot(X_grid, vvKRR_true[:,1], color='black',      linewidth=lw)
    axs[1].plot(X_grid, s2_test,         color='tab:orange',  linewidth=lw)
    axs[1].plot(X_grid, vvKRR_test[:,1], color='tab:blue',    linewidth=lw)
    axs[1].fill_between(X_grid, s2_test - s2_bound, s2_test + s2_bound,
                        color='tab:orange', alpha=0.3)
    axs[1].fill_between(X_grid, vvKRR_test[:,1] - data['vvKRR_bound2'],
                                 vvKRR_test[:,1] + data['vvKRR_bound2'],
                        color='tab:blue', alpha=0.3)
    axs[1].set_ylabel('$f_2(x)$', fontsize=fs)
    axs[1].yaxis.set_label_coords(-0.08, 0.5)
    axs[1].set_xlabel('$x$', fontsize=fs)
    axs[1].set_ylim(-25, 25);  axs[1].set_xlim(0, 1)
    axs[1].grid(True, linewidth=0.4);  axs[1].tick_params(labelsize=fs_tick)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, fontsize=12, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.05), borderpad=0.4,
               handlelength=1.5, columnspacing=1.0)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.28)
    return fig, axs

def plot_vvKRR_uncertainty(vvf, theta, X_train, M_list, yd=2,
                            lambda_=0.01, delta_ub=0.1):
    rng = np.random.default_rng(seed=0)

    Y_train = []
    for xi, Mi in zip(X_train, M_list):
        fx = vvf.get_mean(xi.reshape(1,-1), [np.eye(yd)]).squeeze()
        noise = sample_unit_ball_n(Mi.shape[0], 1, rng).squeeze() * delta_ub
        yi = Mi @ fx + noise
        Y_train.append(yi)

    model = vvKRRPartialNonseparableKernel(
        X_train, M_list, Y_train, theta,
        Tools.Kernel.RBF,
        gamma=vvf.norm, delta_ub=delta_ub,
        bound=np.array([[0, 1]]), lambda_=lambda_)

    X_grid = np.linspace(0, 1, 100).reshape(-1, 1)

    f_true = np.array([vvf.get_mean(x.reshape(1,-1), [np.eye(yd)]).squeeze()
                       for x in X_grid])
    f_pred = np.array([model.get_mean(x.reshape(1,-1), [np.eye(yd)]).squeeze()
                       for x in X_grid])
    bounds = np.array([
        [model.get_bound(x.reshape(1,-1), [np.eye(yd)[[k]]]) for k in range(yd)]
        for x in X_grid
    ])  # (n_grid, yd)

    fig, axs = plt.subplots(1, yd, figsize=(3.5 * yd, 3.0), sharey=False)
    if yd == 1:
        axs = [axs]

    for k, ax in enumerate(axs):
        ax.plot(X_grid.squeeze(), f_true[:, k], color='black',
                linewidth=1.5, label='Ground truth')
        ax.plot(X_grid.squeeze(), f_pred[:, k], color='tab:blue',
                linewidth=1.5, label='vvKRR mean')
        ax.fill_between(X_grid.squeeze(),
                        0 - bounds[:, k],
                        0 + bounds[:, k],
                        color='tab:blue', alpha=0.25, label='Uncertainty')

        for xi, Mi, yi in zip(X_train.reshape(-1), M_list, Y_train):
            for row_idx in range(Mi.shape[0]):
                if np.allclose(Mi[row_idx], np.eye(yd)[k]):
                    ax.scatter(xi, yi[row_idx], color='tab:red', s=30, zorder=5,
                               label='Observation' if k == 0 and row_idx == 0 else '_')

        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel(f'$f_{k+1}(x)$', fontsize=12)
        ax.set_title(f'Dimension {k+1}', fontsize=11)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.4)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, ncol=3,
               loc='upper center', bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.22, wspace=0.3)

    return fig, model

import matplotlib.ticker as mtick

def plot_relative_bound_change_vs_gamma_and_delta(X_train, M_list, Y_train, theta,
                                                    gamma_list, delta_ub_list,
                                                    x_eval=0.5, M_eval=None,
                                                    base_gamma=29.19, base_delta_ub=0.1,
                                                    bound_range=np.array([[0, 1]]), lambda_=0.01):
    """
    Left panel:  relative change in bound at x_eval as gamma varies
                 (delta_ub fixed at base_delta_ub), relative to bound at base_gamma.
                 X-axis shown as % change from base_gamma (0% = base_gamma).
    Right panel: relative change in bound at x_eval as delta_ub varies
                 (gamma fixed at base_gamma), relative to bound at base_delta_ub.
                 X-axis shown as % change from base_delta_ub (0% = base_delta_ub).
    """
    if M_eval is None:
        M_eval = np.eye(2)

    fs = 12; fs_tick = 12; lw = 3.0

    x_eval_arr = np.atleast_2d(x_eval)

    def get_bound_at(gamma, delta_ub):
        model = vvKRRPartialNonseparableKernel(
            X_train, M_list, Y_train, theta,
            Tools.Kernel.RBF,
            gamma=gamma, delta_ub=delta_ub,
            bound=bound_range, lambda_=lambda_
        )
        return np.ravel(model.get_bound(x_eval_arr, [M_eval]))

    # ---- baselines ----
    base_bound_gamma = get_bound_at(base_gamma, base_delta_ub)
    base_bound_delta = get_bound_at(base_gamma, base_delta_ub)  # same model, kept separate for clarity

    # ---- relative change vs gamma ----
    rel_change_gamma = []
    for gamma in gamma_list:
        b = get_bound_at(gamma, base_delta_ub)
        rel_change_gamma.append((b - base_bound_gamma) / base_bound_gamma)
    rel_change_gamma = np.array(rel_change_gamma)

    # ---- relative change vs delta_ub ----
    rel_change_delta = []
    for delta_ub in delta_ub_list:
        b = get_bound_at(base_gamma, delta_ub)
        rel_change_delta.append((b - base_bound_delta) / base_bound_delta)
    rel_change_delta = np.array(rel_change_delta)

    # ---- x-axes as % CHANGE from base value (0% = base) ----
    gamma_pct = (np.asarray(gamma_list) / base_gamma - 1) * 100
    delta_pct = (np.asarray(delta_ub_list) / base_delta_ub - 1) * 100

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 4.0))

    axs[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axs[0].plot(gamma_pct, rel_change_gamma, marker='o', color='tab:blue',
                linewidth=lw, label='Relative bound change')
    axs[0].scatter(0, 0, color='tab:red', s=60, zorder=10, label='True parameter')
    axs[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axs[0].set_xlabel(r'Relative change of $\Gamma$', fontsize=fs)
    axs[0].set_ylabel('Relative change of error bound', fontsize=fs)
    axs[0].set_title(fr'Sensitivity analysis of $\Gamma$', fontsize=fs)
    axs[0].grid(True, linewidth=0.4)
    axs[0].tick_params(labelsize=fs_tick)

    axs[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axs[1].plot(delta_pct, rel_change_delta, marker='o', color='tab:orange',
                linewidth=lw, label='Relative bound change')
    axs[1].scatter(0, 0, color='tab:red', s=60, zorder=10, label='True parameter')
    axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axs[1].set_xlabel('Relative change of noise bound', fontsize=fs)
    axs[1].set_ylabel('Relative change of error bound', fontsize=fs)
    axs[1].set_title(r'Sensitivity analysis of noise bound', fontsize=fs)
    axs[1].grid(True, linewidth=0.4)
    axs[1].tick_params(labelsize=fs_tick)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.32, wspace=0.35)
    return fig, axs

# ── Main ─────────────────────────────────────────────────────────────────────
vvf, theta = get_true_vvf(f_true)

X_train = np.sort(np.random.rand(40))

# data_single = run_scenario(vvf, theta, 'single', X_train)
# data_sum    = run_scenario(vvf, theta, 'sum',    X_train)
# data_grid   = run_scenario(vvf, theta, 'grid')

# fig1, axs1 = plot_two_scenarios(data_single, data_sum,
#                                  titles=['Measurement on $f_1$', 'Measurement on $f_1+f_2$'])
# plt.savefig('1_input_2_output_1.pdf', bbox_inches='tight', dpi=300)

# fig2, axs2 = plot_grid_scenario(data_grid)
# plt.savefig('1_input_2_output_2.pdf', bbox_inches='tight', dpi=300)

X_train_g, Y_train_g, _, M_list_g, *_ = generate_training_data(vvf, 'grid', X_train)

base_gamma = 29.19
base_delta_ub = 0.1

pct_range = np.linspace(70, 500, 15)   # 100% ... 1000%

gamma_list    = base_gamma * pct_range / 100
delta_ub_list = base_delta_ub * pct_range / 100

fig, axs = plot_relative_bound_change_vs_gamma_and_delta(
    X_train_g.reshape(-1, 1), M_list_g, Y_train_g, theta,
    gamma_list=gamma_list,
    delta_ub_list=delta_ub_list,
    x_eval=0.1, M_eval=np.eye(2)[1].reshape(1, -1),
    base_gamma=base_gamma, base_delta_ub=base_delta_ub
)
plt.savefig('1_input_2_output_3.pdf', bbox_inches='tight', dpi=300)

plt.show()