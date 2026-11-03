import numpy as np
import matplotlib.pyplot as plt
from regression.models import KRR, vvKRRPartial, vvKRR
from regression.tools import Tools
from typing import Callable
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors

np.random.seed(0)
def f_true(X:np.ndarray):
    theta = 0.25*np.pi
    Y = np.zeros((np.shape(X)[0],2))
    A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    D = np.array([[1,0],[0,0.01]])
    T = A@D@A.T
    V = np.column_stack((np.cos(2*np.pi*X)*5, np.sin(2*np.pi*X)*5))
    Y = V@T.T
    return Y, T

def get_true_vvf(f_true:Callable):
    X_train = np.linspace(0,1,50)
    Y_train,T = f_true(X_train)
    X_train = np.reshape(X_train,(-1,1))

    vvf = vvKRR(X_train, Y_train, Tools.Kernel.RBF, T, length_scale = 0.1, gamma = 5, delta_ub = 0.1, bound = np.array([[0,1]]), lambda_ = 0.01)

    return vvf, T

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
        fx = vvf.get_mean(x)

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

def run_scenario(vvf, T, scenario_type, X_train=None):
    (X_train, Y_train, Y_train_full, M_list, meas_type,
     X_train_scalar_1, Y_train_scalar_1,
     X_train_scalar_2, Y_train_scalar_2,
     X_train_scalar_sum, Y_train_scalar_sum) = generate_training_data(vvf, scenario_type, X_train)

    vvKRR_model = vvKRRPartial(X_train.reshape(-1,1), M_list, Y_train,
                                Tools.Kernel.RBF, T=T, length_scale=0.1,
                                gamma=vvf.norm, delta_ub=0.1,
                                bound=np.array([[0,1]]), lambda_=0.01)

    scalarKRR1 = KRR(X_train_scalar_1.reshape(-1,1), Y_train_scalar_1,
                     Tools.Kernel.RBF, length_scale=0.1, gamma=vvf.norm,
                     delta_ub=0.1, bound=np.array([[0,1]]), lambda_=0.01)

    scalarKRR2 = KRR(X_train_scalar_2.reshape(-1,1), Y_train_scalar_2,
                     Tools.Kernel.RBF, length_scale=0.1, gamma=vvf.norm,
                     delta_ub=0.1, bound=np.array([[0,1]]), lambda_=0.001)

    X_grid = np.linspace(0, 1, 100)
    vvKRR_true     = np.array([vvf.get_mean(np.atleast_2d(i))                                   for i in X_grid])
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
    color_map = {'full': 'black', 'first': 'red', 'second': 'red', 'sum': 'red'}

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
    lw = 3.0;  ms = 25;  fs = 12;  fs_tick = 12

    # ---- f1 ----
    ax = axs_col[0]
    mask1 = np.array([t in ('full', 'first') for t in meas_type])
    ax.scatter(X_train[mask1], Y_train_full[mask1, 0],
               c=[point_colors[i] for i in np.where(mask1)[0]],
               edgecolors='white', label='Sample', linewidths=1.5, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,0], color='black',      linewidth=lw, label='Ground truth')
    ax.plot(X_grid, s1_test,         color='tab:orange',  linewidth=lw, label='Scalar-valued KRR')
    ax.plot(X_grid, vvKRR_test[:,0], color='tab:blue',    linewidth=lw, label='Vector-valued KRR')
    ax.fill_between(X_grid, s1_test - s1_bound, s1_test + s1_bound,
                    color='tab:grey', alpha=0.3, label='Scalar-valued KRR bound')
    ax.fill_between(X_grid, vvKRR_test[:,0] - data['vvKRR_bound1'],
                             vvKRR_test[:,0] + data['vvKRR_bound1'],
                    color='tab:blue', alpha=0.3, label='Vector-valued KRR bound')
    if show_ylabel:
        ax.set_ylabel('$f_1(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_ylim(-10, 18);  ax.grid(True, linewidth=0.4);  ax.tick_params(labelsize=fs_tick)

    # ---- f2 ----
    ax = axs_col[1]
    mask2 = np.array([t in ('full', 'second', 'sum') for t in meas_type])
    ax.scatter(X_train[mask2], Y_train_full[mask2, 1],
               c=[point_colors[i] for i in np.where(mask2)[0]],
               edgecolors='white', linewidths=1.5, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,1], color='black',      linewidth=lw)
    ax.plot(X_grid, s2_test,         color='tab:orange',  linewidth=lw)
    ax.plot(X_grid, vvKRR_test[:,1], color='tab:blue',    linewidth=lw)
    ax.fill_between(X_grid, s2_test - s2_bound, s2_test + s2_bound,
                    color='tab:grey', alpha=0.3)
    ax.fill_between(X_grid, vvKRR_test[:,1] - data['vvKRR_bound2'],
                             vvKRR_test[:,1] + data['vvKRR_bound2'],
                    color='tab:blue', alpha=0.3)
    if show_ylabel:
        ax.set_ylabel('$f_2(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_ylim(-10, 18);  ax.grid(True, linewidth=0.4);  ax.tick_params(labelsize=fs_tick)

    # ---- f1 + f2 ----
    ax = axs_col[2]
    mask_sum = np.array([t in ('full', 'sum') for t in meas_type])
    ax.scatter(X_train[mask_sum], Y_train_full[mask_sum, 2],
               c=[point_colors[i] for i in np.where(mask_sum)[0]],
               edgecolors='white', linewidths=1.5, s=ms, zorder=10)
    ax.plot(X_grid, vvKRR_true[:,0] + vvKRR_true[:,1], color='black',      linewidth=lw)
    ax.plot(X_grid, s1_test + s2_test,                  color='tab:orange',  linewidth=lw)
    ax.plot(X_grid, vvKRR_test[:,0] + vvKRR_test[:,1],  color='tab:blue',    linewidth=lw)
    ax.fill_between(X_grid,
                    (s1_test + s2_test) - (s1_bound + s2_bound),
                    (s1_test + s2_test) + (s1_bound + s2_bound),
                    color='tab:grey', alpha=0.3)
    ax.fill_between(X_grid,
                    (vvKRR_test[:,0] + vvKRR_test[:,1]) - data['vvKRR_boundsum'],
                    (vvKRR_test[:,0] + vvKRR_test[:,1]) + data['vvKRR_boundsum'],
                    color='tab:blue', alpha=0.3)
    if show_ylabel:
        ax.set_ylabel('$f_1(x){+}f_2(x)$', fontsize=fs)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_xlabel('$x$', fontsize=fs)
    ax.set_ylim(-10, 18);  ax.set_xlim(0, 0.5)
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
               bbox_to_anchor=(0.5, 0.1), borderpad=0.4,
               handlelength=1.5, columnspacing=1.0)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.28)
    return fig, axs


def plot_grid_scenario(data):
    lw = 3.0;  ms = 25;  fs = 12;  fs_tick = 12
    color_map    = {'full': 'black', 'first': 'red', 'second': 'orange', 'sum': 'black'}
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
                   edgecolors='white', linewidths=1.5, s=ms, zorder=10)
    axs[0].plot(X_grid, vvKRR_true[:,0], color='black',      linewidth=lw, label='Ground truth')
    axs[0].plot(X_grid, s1_test,         color='tab:orange',  linewidth=lw, label='Scalar-valued KRR')
    axs[0].plot(X_grid, vvKRR_test[:,0], color='tab:blue',    linewidth=lw, label='Vector-valued KRR')
    axs[0].fill_between(X_grid, s1_test - s1_bound, s1_test + s1_bound,
                        color='tab:grey', alpha=0.3, label='Scalar-valued KRR bound')
    axs[0].fill_between(X_grid, vvKRR_test[:,0] - data['vvKRR_bound1'],
                                 vvKRR_test[:,0] + data['vvKRR_bound1'],
                        color='tab:blue', alpha=0.3, label='Vector-valued KRR bound')
    axs[0].set_ylabel('$f_1(x)$', fontsize=fs)
    axs[0].yaxis.set_label_coords(-0.08, 0.5)
    axs[0].set_ylim(-5, 5);  axs[0].grid(True, linewidth=0.4);  axs[0].tick_params(labelsize=fs_tick)

    # ---- zoomed inset on f1 (lower-left of axs[0]) ----
    x1, x2, y1, y2 = 0.35, 0.4, -1.3, 1.3          # region to zoom into
    ax_inset = axs[0].inset_axes(
        [0.05, 0.05, 0.15, 0.6],                    # [left, bottom, width, height] in axes fraction
        xlim=(x1, x2), ylim=(y1, y2)
    )
    # re-plot everything on the inset
    ax_inset.scatter(X_train_g[mask1], Y_train_g[mask1, 0],
                     c=[point_colors[i] for i in np.where(mask1)[0]],
                     edgecolors='white', linewidths=1.5, s=ms, zorder=10)
    ax_inset.plot(X_grid, vvKRR_true[:,0],  color='black',     linewidth=lw)
    ax_inset.plot(X_grid, s1_test,          color='tab:orange', linewidth=lw)
    ax_inset.plot(X_grid, vvKRR_test[:,0],  color='tab:blue',   linewidth=lw)
    ax_inset.fill_between(X_grid, s1_test - s1_bound, s1_test + s1_bound,
                          color='tab:grey', alpha=0.3)
    ax_inset.fill_between(X_grid,
                          vvKRR_test[:,0] - data['vvKRR_bound1'],
                          vvKRR_test[:,0] + data['vvKRR_bound1'],
                          color='tab:blue', alpha=0.3)
    ax_inset.set_xlim(x1, x2);  ax_inset.set_ylim(y1+0.1, y2-0.1)
    ax_inset.tick_params(labelbottom=False, labelleft=False)

    # draw zoom box + connector lines
    axs[0].indicate_inset_zoom(ax_inset, edgecolor='black', linewidth=1.0)
    # ---- f2 ----
    mask2 = np.array([t in ('full', 'second') for t in meas_type])
    axs[1].scatter(X_train_g[mask2], Y_train_g[mask2, 1],
                   c=[point_colors[i] for i in np.where(mask2)[0]],
                   edgecolors='white', linewidths=1.5, s=ms, zorder=10)
    axs[1].plot(X_grid, vvKRR_true[:,1], color='black',      linewidth=lw)
    axs[1].plot(X_grid, s2_test,         color='tab:orange',  linewidth=lw)
    axs[1].plot(X_grid, vvKRR_test[:,1], color='tab:blue',    linewidth=lw)
    axs[1].fill_between(X_grid, s2_test - s2_bound, s2_test + s2_bound,
                        color='tab:grey', alpha=0.3)
    axs[1].fill_between(X_grid, vvKRR_test[:,1] - data['vvKRR_bound2'],
                                 vvKRR_test[:,1] + data['vvKRR_bound2'],
                        color='tab:blue', alpha=0.3)
    axs[1].set_ylabel('$f_2(x)$', fontsize=fs)
    axs[1].yaxis.set_label_coords(-0.08, 0.5)
    axs[1].set_xlabel('$x$', fontsize=fs)
    axs[1].set_ylim(-5, 5);  axs[1].set_xlim(0, 1)
    axs[1].grid(True, linewidth=0.4);  axs[1].tick_params(labelsize=fs_tick)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3, fontsize=12, framealpha=0.8,
               bbox_to_anchor=(0.5, -0.0), borderpad=0.4,
               handlelength=1.5, columnspacing=1.0)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.28)
    return fig, axs


# ── Main ─────────────────────────────────────────────────────────────────────
vvf, T = get_true_vvf(f_true)
X_train = np.sort(np.random.rand(40))

data_single = run_scenario(vvf, T, 'single', X_train)
data_sum    = run_scenario(vvf, T, 'sum',    X_train)
data_grid   = run_scenario(vvf, T, 'grid')

fig1, axs1 = plot_two_scenarios(data_single, data_sum,
                                 titles=['$M_7 = [1,0]$', '$M_7 = [1,1]$'])
# plt.savefig('1_input_2_output_1.pdf', bbox_inches='tight', dpi=300)

fig2, axs2 = plot_grid_scenario(data_grid)
# plt.savefig('1_input_2_output_2.pdf', bbox_inches='tight', dpi=300)

plt.show()