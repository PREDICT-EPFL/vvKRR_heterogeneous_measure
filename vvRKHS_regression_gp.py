import numpy as np
import matplotlib.pyplot as plt
from regression.models import vvKRRPartial
from regression.tools import Tools
from typing import Callable
from matplotlib.lines import Line2D

np.random.seed(0)
def f_true(X:np.ndarray):
    '''
    This ture function returns alpha, not function value
    '''
    rng      = np.random.default_rng(seed=0)
    alpha    = rng.uniform(-1, 1, (5, 101))
    x_node = np.linspace(0,1,5)
    t_node = np.linspace(0,1,101)
    KT = Tools.Kernel.RBF(t_node.reshape(-1,1),length_scale = 0.2)  + 0.00001*np.eye(101)
    KT_inv = np.linalg.inv(KT)
    KT_integral = compute_kernel_gram_integral(t_node, lambda x, y: Tools.Kernel.RBF(x, y, length_scale=0.2)) + 1e-8*np.eye(len(t_node))
    
    KT_eigvalue, KT_eigvector = np.linalg.eigh(KT)
    KT_half = KT_eigvector@np.sqrt(np.diag(KT_eigvalue))@KT_eigvector.T
    KT_half_inv = KT_eigvector@np.sqrt(np.diag(1/KT_eigvalue))@KT_eigvector.T
    Lam_T = np.diag(pow(10,np.linspace(-10,0,101)))
    B = KT_eigvector@Lam_T@KT_eigvector.T

    G_xX = Tools.Kernel.RBF(X, x_node.reshape(-1,1),length_scale = 0.2)
    G = Tools.Kernel.RBF(x_node.reshape(-1,1),length_scale = 0.2)
    G_inv = np.linalg.inv(G)

    T = np.eye(101)

    return np.squeeze(G_xX@G_inv@alpha), T, KT, KT_inv, KT_half, KT_integral, t_node

def get_true_vvf(f_true:Callable):
    X_train = Tools.others.make_grid(np.array([[0,1]]),4)
    Y_train, T, KT, KT_inv, KT_half, KT_integral, t_node = f_true(X_train)
    
    M_list = [np.eye(101) for i in range(np.shape(X_train)[0])]
    vvf = vvKRRPartial(X_train, M_list, Y_train@KT_half, Tools.Kernel.RBF, T, length_scale = 0.2, gamma = 5, delta_ub = 0.1, bound = np.array([[0,1]]), lambda_ = 0.01, if_delta= False)
    
    return vvf, T, KT, KT_inv, KT_half, KT_integral, t_node

def sample_unit_ball_n(N, n_samples,rng):
    z = rng.normal(size=(n_samples, N))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / norms
    
    r = rng.random(n_samples) ** (1.0 / N)
    r = r.reshape(-1, 1)
    
    return r * u

def compute_kernel_gram_integral(t_nodes, kernel_fn, n_int=1000):
    """
    Computes M_{ij} = integral k(t, t_i) k(t, t_j) dt over [0,1].
    t_nodes: (p,) array of output nodes
    Returns: (p, p) matrix
    """
    t_int = np.linspace(0, 1, n_int).reshape(-1, 1)  # integration grid (n_int, 1)
    dt = 1.0 / (n_int - 1)

    # K[k, i] = k(t_int[k], t_nodes[i]),  shape (n_int, p)
    K = kernel_fn(t_int, t_nodes.reshape(-1, 1))

    # M = K^T K * dt,  shape (p, p)
    return K.T @ K * dt

def generate_training_data(vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=0.01, n_meas=5,
                           meas_mode='point', special_meas=None,
                           n_train=5, x_sample='grid'):
    rng = np.random.default_rng(seed=0)

    yd = 101

    if x_sample == 'grid':
        X_train = np.linspace(0, 1, n_train).reshape(-1, 1)
    elif x_sample == 'random':
        X_train = np.sort(rng.random(n_train)).reshape(-1, 1)
    else:
        X_train = np.sort(x_sample).reshape(-1, 1)

    num_sample = X_train.shape[0]

    M_list = []
    Y_train = []
    meas_type = []
    Y_train_full = np.full((num_sample, yd), np.nan)

    def make_integral_Mi():
        weights = rng.uniform(-1, 1, (n_meas, yd))
        return weights@KT_integral@KT_inv@KT_half

    def make_integral_Mi_coords(coords):
        # one measurement row: 1s at coords, 0s elsewhere
        Mi = np.zeros((1, yd))
        Mi[0, list(coords)] = 1.0
        return Mi@KT_integral@KT_inv@KT_half

    # --- regular samples ---
    for i, v in enumerate(X_train):
        x        = np.atleast_2d(v)
        z = vvf.get_mean(x, [np.eye(yd)]).squeeze()

        if meas_mode == 'full':
            coords  = np.arange(yd)
            Mi      = np.eye(yd)
            noise_i = sample_unit_ball_n(yd, 1, rng).squeeze() * delta_ub
            yi = Mi @ z + noise_i
        elif meas_mode == 'point':
            coords  = np.sort(rng.choice(yd, size=n_meas, replace=False))
            Mi      = np.eye(yd)[coords]@KT_half
            noise_i = sample_unit_ball_n(Mi.shape[0], 1, rng).squeeze() * delta_ub
            yi = Mi @ z + noise_i
        elif meas_mode == 'integral':
            coords  = ()
            Mi      = make_integral_Mi()
            noise_i = sample_unit_ball_n(Mi.shape[0], 1, rng).squeeze() * delta_ub
            yi = Mi @ z + noise_i
        else:
            raise ValueError(f"Unknown meas_mode: {meas_mode}")

        M_list.append(Mi)
        Y_train.append(yi)
        meas_type.append((meas_mode, coords if meas_mode != 'integral' else ()))
        if meas_mode == 'point':
            Y_train_full[i, coords] = yi

    # --- additional special samples ---
    if special_meas is not None:
        for spec in special_meas:
            x        = np.atleast_2d(spec['x'])
            mode_i   = spec['mode']
            z = vvf.get_mean(x, [np.eye(yd)]).squeeze()

            if mode_i == 'full':
                coords  = np.arange(yd)
                Mi      = np.eye(yd)
                noise_i = sample_unit_ball_n(yd, 1, rng).squeeze() * delta_ub
                yi = Mi @ z + noise_i
            elif mode_i == 'point':
                coords  = np.sort(np.array(spec['coords'])) if 'coords' in spec \
                          else np.sort(rng.choice(yd, size=n_meas, replace=False))
                Mi      = np.eye(yd)[coords] @ KT_half
                noise_i = sample_unit_ball_n(Mi.shape[0], 1, rng).squeeze() * delta_ub
                yi = Mi @ z + noise_i
            elif mode_i == 'integral':
                if 'coords' in spec:
                    coords  = np.sort(np.array(spec['coords']))
                    Mi      = make_integral_Mi_coords(coords)
                else:
                    coords  = ()
                    Mi      = make_integral_Mi()
                noise_i = sample_unit_ball_n(Mi.shape[0], 1, rng).squeeze() * delta_ub
                yi = Mi @ z + noise_i
            else:
                raise ValueError(f"Unknown mode: {mode_i}")

            X_train = np.vstack([X_train, x])
            M_list.append(Mi)
            Y_train.append(yi)
            meas_type.append((mode_i, coords if mode_i != 'integral' else coords))

    return X_train, Y_train, Y_train_full, M_list, meas_type

def comparison_norm_uncertainty(vvf, t_node, KT, KT_inv, KT_half, KT_integral, T,
                                 n_meas=5, lambda_=0.01, delta_ub=0.01):
    '''
    plot the uncertainty of \|f(x)\| for each x
    '''
    yd     = len(t_node)
    X_grid = np.linspace(0, 1, 50).reshape(-1, 1)

    # --- generate data and train models ---
    special_meas = [
        {'x': 0.5, 'mode': 'point'}
    ]
    X_train, Y_train, _, M_list, meas_type = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub, special_meas=None, n_meas=50, meas_mode='point',n_train = 3,x_sample = 'grid')

    vvKRR_partial = vvKRRPartial(X_train, M_list, Y_train,
                                               Tools.Kernel.RBF, T,
                                               length_scale=0.2,
                                               gamma=vvf.norm, delta_ub=delta_ub,
                                               bound=np.array([[0, 1]]),
                                               lambda_=lambda_,
                                               if_delta=True,
                                               numerical_lambda=8)
    
    # --- evaluate over grid ---
    norm_true    = np.array([np.linalg.norm(vvf.get_mean(x.reshape(1,-1), [np.eye(yd)]).squeeze())
                             for x in X_grid])
    norm_partial = np.array([np.linalg.norm(vvKRR_partial.get_mean(x.reshape(1,-1), [np.eye(yd)]).squeeze())
                             for x in X_grid])
    # norm_full = np.array([np.sqrt(vvKRR_full.get_mean(x.reshape(1,-1)).squeeze()@T@vvKRR_full.get_mean(x.reshape(1,-1)).squeeze())
    #                          for x in X_grid])

    # bound on norm: | ||f|| - ||mu|| | <= ||f - mu|| <= b
    b_partial = np.array([vvKRR_partial.get_bound(x.reshape(1,-1), [np.eye(yd)]) for x in X_grid])
    # b_full = np.array([vvKRR_full.get_bound(x.reshape(1,-1)) for x in X_grid])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(X_grid.squeeze(), norm_true,    color='black',      linewidth=2.0, label='Ground truth')
    ax.plot(X_grid.squeeze(), norm_partial, color='tab:blue',   linewidth=2.0, label='vvKRR (partial)')
    # ax.plot(X_grid.squeeze(), norm_partial, color='tab:orange',   linewidth=2.0, label='vvKRR (full)')
    ax.fill_between(X_grid.squeeze(),
                    norm_partial - b_partial, norm_partial + b_partial,
                    color='tab:blue',   alpha=0.3)
    # ax.fill_between(X_grid.squeeze(),
    #                 0 - b_full, 0 + b_full,
    #                 color='tab:orange',   alpha=0.3)

    ax.set_xlabel('$x$',          fontsize=12)
    ax.set_ylabel('$\|f(x)\|$',   fontsize=12)
    ax.set_title('Norm uncertainty', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    return fig, ax

def plot_3d_prediction_and_error(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral,
                       n_train=5, n_meas=5, meas_mode='point',
                       lambda_=0.01, delta_ub=0.01,
                       plot_bound=False):
    yd = len(t_node)
    X_grid = np.linspace(0, 1, 50).reshape(-1, 1)

    special_meas = [
        {'x': 0.0, 'mode': 'full'},
        {'x': 0.5, 'mode': 'full'},
        {'x': 1.0, 'mode': 'full'},
    ]

    X_train, Y_train, _, M_list, meas_type = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode=meas_mode, special_meas=special_meas,
        n_train=n_train, x_sample=[0.2, 0.3, 0.6, 0.75, 0.9])

    vvKRR_model = vvKRRPartial(
        X_train, M_list, Y_train,
        Tools.Kernel.RBF, T,
        length_scale=0.2,
        gamma=vvf.norm, delta_ub=delta_ub,
        bound=np.array([[0, 1]]),
        lambda_=lambda_,
        if_delta=True,
        numerical_lambda=8)

    vvKRR_pred  = np.array([vvKRR_model.get_mean(np.atleast_2d(x), [np.eye(yd)]) @ KT_half for x in X_grid])
    vvKRR_true  = np.array([vvf.get_mean(np.atleast_2d(x), [np.eye(yd)]) @ KT_half for x in X_grid])
    vvKRR_error = vvKRR_pred - vvKRR_true

    if plot_bound:
        vvKRR_bound = np.array([
            [vvKRR_model.get_bound(np.atleast_2d(x), [np.eye(yd)[[k]] @ KT_half]) for k in range(yd)]
            for x in X_grid
        ])

    XX, TT = np.meshgrid(X_grid.squeeze(), t_node)

    fig = plt.figure(figsize=(7.16, 3.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # ── Left: prediction ──────────────────────────────────────────────────────
    ax_pred   = fig.add_subplot(gs[0], projection='3d')
    surf_pred = ax_pred.plot_surface(XX, TT, vvKRR_pred.T, cmap='viridis', alpha=0.7)

    if plot_bound:
        ax_pred.plot_surface(XX, TT, vvKRR_pred.T + vvKRR_bound.T,
                             color='tab:grey', alpha=0.3, zorder=3)
        ax_pred.plot_surface(XX, TT, vvKRR_pred.T - vvKRR_bound.T,
                             color='tab:grey', alpha=0.3, zorder=1)

    for xi, (mode_i, coords), yi in zip(X_train.reshape(-1), meas_type, Y_train):
        if mode_i == 'full':
            ax_pred.plot(np.full_like(t_node, xi), t_node, KT_half @ yi,
                         color='red', linewidth=3.0, zorder=5)
        elif mode_i == 'point':
            t_coords = t_node[list(coords)]
            ax_pred.scatter(np.full_like(t_coords, xi), t_coords, yi,
                            color='red', s=10, zorder=5, depthshade=False)

    ax_pred.set_xlabel('$x$'); ax_pred.set_ylabel('$t$')
    ax_pred.set_title('Prediction')
    ax_pred.set_zlim(-4, 8)
    ax_pred.set_box_aspect([1, 1, 0.9])
    ax_pred.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_pred.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax_pred.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax_pred.zaxis.set_tick_params(labelright=False, labelleft=True)

    cbar_pred = fig.colorbar(surf_pred, ax=ax_pred, shrink=0.5, pad=0.1, aspect=15)
    # cbar_pred.set_label('$f(x,t)$', fontsize=9)
    cbar_pred.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color='red', linewidth=2.0, label='Full obs.'),
        Line2D([0], [0], color='red', linewidth=0, marker='o', markersize=4, label='Point obs.'),
    ]
    ax_pred.legend(handles=legend_handles, fontsize=8,
                   loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # ── Right: error ──────────────────────────────────────────────────────────
    ax_err      = fig.add_subplot(gs[1], projection='3d')
    err_abs_max = np.abs(vvKRR_error).max()
    surf_err    = ax_err.plot_surface(XX, TT, vvKRR_error.T, cmap='viridis',
                                      vmin=-err_abs_max, vmax=err_abs_max, alpha=0.8)

    ax_err.set_xlabel('$x$'); ax_err.set_ylabel('$t$')
    ax_err.set_title('Prediction error')
    ax_err.set_zlim(-4, 8)
    ax_err.set_box_aspect([1, 1, 0.9])
    ax_err.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_err.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax_err.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax_err.zaxis.set_tick_params(labelright=False, labelleft=True)

    cbar_err = fig.colorbar(surf_err, ax=ax_err, shrink=0.5, pad=0.1, aspect=15)
    # cbar_err.set_label('Error', fontsize=9)
    cbar_err.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig

def comparison_full_point(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral, 
                                x_queries,
                                n_train_point=5, n_train_full=3,
                                n_meas=5,
                                lambda_=0.01, delta_ub=0.01):
    yd = len(t_node)

    special_meas_full = [
        {'x': 0.0, 'mode': 'full'},
        {'x': 0.5, 'mode': 'full'},
        {'x': 1.0, 'mode': 'full'},
    ]

    X_pt, Y_pt, _, M_pt, mt_pt = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='point',
        n_train=n_train_point, x_sample='grid')

    X_fu, Y_fu, _, M_fu, mt_fu = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='full',
        n_train=n_train_full, x_sample='grid')

    X_all, Y_all, _, M_all, mt_all = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='point', special_meas=special_meas_full,
        n_train=n_train_point, x_sample='grid')

    def make_model(X, M, Y):
        return vvKRRPartial(
            X, M, Y, Tools.Kernel.RBF, T,
            length_scale=0.2, gamma=vvf.norm, delta_ub=delta_ub,
            bound=np.array([[0, 1]]), lambda_=lambda_, if_delta=True)

    model_pt  = make_model(X_pt,  M_pt,  Y_pt)
    model_fu  = make_model(X_fu,  M_fu,  Y_fu)
    model_all = make_model(X_all, M_all, Y_all)

    models_cfg = [
        (model_fu,  'tab:orange', r'vvKRR model with $\mathcal{D}_{\mathrm{full}}$'),
        (model_pt,  'tab:red',    r'vvKRR model with $\mathcal{D}_{\mathrm{point}}$'),
        (model_all, 'tab:blue',   r'vvKRR model with $\mathcal{D}_{\mathrm{point}}\cup\mathcal{D}_{\mathrm{full}}$'),
    ]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.16, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)
    axs = np.array([[ax00, ax01], [ax10, ax11]])
    axs_flat = axs.flatten()

    for idx, (ax, xq) in enumerate(zip(axs_flat, x_queries)):
        x = np.atleast_2d(xq)
        f_true = KT_half @ vvf.get_mean(x, [np.eye(yd)]).squeeze()
        ax.plot(t_node, f_true, color='black', linewidth=1.5,
                label='Ground truth' if idx == 0 else '_')

        for model, color, label in models_cfg:
            f_mean = KT_half @ model.get_mean(x, [np.eye(yd)]).squeeze()
            bound  = np.array([model.get_bound(x, [np.eye(yd)[[k]] @ KT_half]) for k in range(yd)])
            ax.plot(t_node, f_mean, color=color, linewidth=1.5,
                    label=label if idx == 0 else '_')
            ax.fill_between(t_node, f_mean - bound, f_mean + bound,
                            color=color, alpha=0.25)

        # title as upper-right text inside the axes
        ax.text(0.97, 0.97, f'$x = {float(xq):.2f}$',
                transform=ax.transAxes, fontsize=12,
                ha='right', va='top')

        if idx % 2 == 0:
            ax.set_ylabel('$f(x)(t)$', fontsize=12)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.4)

    axs[0, 0].set_ylim(-4, 10)
    axs[1, 0].set_ylim(-8, 14)

    for ax in axs[1]:
        ax.set_xlabel('$t$', fontsize=12)
    for ax in axs[0]:
        ax.tick_params(labelbottom=False)
    for ax in [ax01, ax11]:
        ax.tick_params(labelleft=False)
    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, ncol=2,
               loc='upper center', bbox_to_anchor=(0.5, 0.09))
    fig.subplots_adjust(bottom=0.18)

    return fig

def comparison_integral_point_1(vvf, t_node, T, KT_inv, KT_half, KT_integral,
                               x_query, x_meas,
                               point_coord, integral_coords,
                               lambda_=0.01, delta_ub=0.01):
    yd = len(t_node)

    special_point    = [{'x': x_meas, 'mode': 'point',    'coords': point_coord}]
    special_integral = [{'x': x_meas, 'mode': 'integral', 'coords': integral_coords}]

    def make_model(special_meas):
        X_train, Y_train, _, M_list, meas_type = generate_training_data(
            vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
            n_meas=1, meas_mode='point', special_meas=special_meas,
            n_train=0, x_sample='grid')
        model = vvKRRPartial(
            X_train, M_list, Y_train,
            Tools.Kernel.RBF, T,
            length_scale=0.2, gamma=vvf.norm, delta_ub=delta_ub,
            bound=np.array([[0, 1]]), lambda_=lambda_, if_delta=True)
        return model, Y_train, M_list

    model_point,    Y_pt, M_pt = make_model(special_point)
    model_integral, Y_in, M_in = make_model(special_integral)

    x      = np.atleast_2d(x_query)
    f_true = KT_half @ vvf.get_mean(x, [np.eye(yd)]).squeeze()

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
    configs = [
        (model_point,    axs[0], f'x={x_query}'),
        (model_integral, axs[1], f'x={x_query}'),
    ]

    for model, ax, title in configs:
        f_mean = KT_half @ model.get_mean(x, [np.eye(yd)]).squeeze()
        bound  = np.array([model.get_bound(x, [np.eye(yd)[[k]]@KT_half]) for k in range(yd)])

        ax.plot(t_node, f_true, color='black',    linewidth=1.5, label='Ground truth')
        ax.plot(t_node, f_mean, color='tab:blue', linewidth=1.5, label='Vector-valued model')
        ax.fill_between(t_node, f_mean - bound, f_mean + bound,
                        color='tab:blue', alpha=0.25)

        ax.text(0.97, 0.97, title, transform=ax.transAxes, fontsize=12,
                ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='none', pad=3))
        ax.set_xlabel('$t$', fontsize=12)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.4)
        ax.set_ylim(-20,24)

    # ── Left: scatter point measurements ─────────────────────────────────────
    t_coords = t_node[list(point_coord)]
    y_coords = Y_pt[0]  # only one special sample, shape (len(point_coord),)
    axs[0].scatter(t_coords, y_coords, color='tab:red', s=40, zorder=5,
                   label='Point evaluation')

    # ── Right: plot integral indicator function ───────────────────────────────
    Mi_integral = M_in[0]  # shape (1, yd)
    axs[1].plot(t_node, 40*(Mi_integral@KT_half).squeeze(), color='tab:red', linewidth=1.5,
                linestyle='--', label='Integral function (scaled)')

    axs[0].set_ylabel('$f(x)(t)$', fontsize=12)
    axs[1].tick_params(labelleft=False)

    # rebuild legend after adding scatter/line
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        if ax == axs[0]:
            handles, labels = h, l

    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()
    # merge unique labels
    seen = {}
    for h, l in zip(handles0 + handles1, labels0 + labels1):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), fontsize=12, ncol=2,
               loc='upper center', bbox_to_anchor=(0.5, 0.04))
    fig.subplots_adjust(bottom=0.22, wspace=0.2)

    return fig

def comparison_integral_point_2(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral,
                               x_query,
                               spec_base, spec_1, spec_2,
                               lambda_=0.01, delta_ub=0.01):
    yd = len(t_node)

    def make_model(special_meas):
        X_train, Y_train, _, M_list, meas_type = generate_training_data(
            vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
            n_meas=3, meas_mode='point', special_meas=special_meas,
            n_train=0, x_sample='grid')
        model = vvKRRPartial(
            X_train, M_list, Y_train,
            Tools.Kernel.RBF, T,
            length_scale=0.2, gamma=vvf.norm, delta_ub=delta_ub,
            bound=np.array([[0, 1]]), lambda_=lambda_, if_delta=True)
        return model, Y_train, meas_type

    model_base, Y_base, mt_base = make_model(spec_base)
    model_1,    Y_1,    mt_1    = make_model(spec_base + spec_1)
    model_2,    Y_2,    mt_2    = make_model(spec_base + spec_2)

    x      = np.atleast_2d(x_query)
    f_true = KT_half @ vvf.get_mean(x, [np.eye(yd)]).squeeze()

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)

    plot_configs = [
        (axs[0], model_base, model_1, mt_1, Y_1, 'One integral measurement'),
        (axs[1], model_base, model_2, mt_2, Y_2, 'Five integral measurements'),
    ]

    for col, (ax, m_base, m_extra, mt, Y_train, extra_label) in enumerate(plot_configs):
        ax.plot(t_node, f_true, color='black', linewidth=1.5,
                label='Ground truth' if col == 0 else '_')

        for model, color, label in [
            (m_base,  'tab:orange', 'vvKRR with point measurement'),
            (m_extra, 'tab:blue',   'vvKRR with mixed measurement'),
        ]:
            f_mean = KT_half @ model.get_mean(x, [np.eye(yd)]).squeeze()
            bound  = np.array([model.get_bound(x, [np.eye(yd)[[k]] @ KT_half]) for k in range(yd)])
            ax.plot(t_node, f_mean, color=color, linewidth=1.5,
                    label=label if col == 0 else '_')
            ax.fill_between(t_node, f_mean - bound, f_mean + bound,
                            color=color, alpha=0.25)

        # scatter point measurements
        for (mode_i, coords), yi in zip(mt, Y_train):
            if mode_i == 'point' and len(coords) > 0:
                t_coords = t_node[list(coords)]
                ax.scatter(t_coords, yi, color='tab:red', s=20, zorder=5,
                           label='Point evaluation' if col == 0 else '_')

        ax.set_title(extra_label, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_xlabel('$t$', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.4)

    axs[0].set_ylabel('$f(x)(t)$', fontsize=12)
    axs[1].tick_params(labelleft=False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, ncol=2,
               loc='upper center', bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(bottom=0.22, wspace=0.2)

    return fig

def comparison_four_models(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral,
                           n_meas=5, n_train=5, lambda_=0.01, delta_ub=0.01):
    yd     = len(t_node)
    X_grid = np.linspace(0, 1, 50).reshape(-1, 1)
    rng    = np.random.default_rng(seed=42)

    coeff   = rng.uniform(-1, 1, yd)
    coeff  /= np.linalg.norm(coeff)
    M_query = [coeff.reshape(1, -1) @ KT_integral @ KT_inv @ KT_half]

    def make_model(X, M, Y):
        return vvKRRPartial(
            X, M, Y, Tools.Kernel.RBF, T,
            length_scale=0.2, gamma=vvf.norm, delta_ub=delta_ub,
            bound=np.array([[0, 1]]), lambda_=lambda_, if_delta=True)

    # --- generate data ---
    X_fu, Y_fu, _, M_fu, _ = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='full', n_train=3, x_sample='grid')

    X_pt, Y_pt, _, M_pt, _ = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='point', n_train=n_train, x_sample=[0.2,0.3,0.6,0.75,0.9])

    X_in, Y_in, _, M_in, _ = generate_training_data(
        vvf, T, KT, KT_inv, KT_half, KT_integral, delta_ub=delta_ub,
        n_meas=n_meas, meas_mode='integral', n_train=n_train, x_sample=[0.1,0.25, 0.4, 0.7, 0.8])

    X_all = np.vstack([X_fu, X_pt, X_in])
    Y_all = Y_fu + Y_pt + Y_in
    M_all = M_fu + M_pt + M_in

    row_configs = [
        (make_model(X_fu,  M_fu,  Y_fu),  'tab:orange', r'$\mathcal{D}_{\mathrm{full}}$'),
        (make_model(X_pt,  M_pt,  Y_pt),  'tab:red',    r'$\mathcal{D}_{\mathrm{point}}$'),
        (make_model(X_in,  M_in,  Y_in),  'tab:green',  r'$\mathcal{D}_{\mathrm{integral}}$'),
        (make_model(X_all, M_all, Y_all), 'tab:blue',   'Combined'),
    ]

    # --- ground truth ---
    norm_true = np.array([
        np.linalg.norm(vvf.get_mean(x.reshape(1, -1), [np.eye(yd) @ KT_half]).squeeze())
        for x in X_grid])
    intg_true = np.array([
        float(vvf.get_mean(x.reshape(1, -1), M_query))
        for x in X_grid])

    # --- figure: 4 rows x 2 cols, same total size ---
    fig = plt.figure(figsize=(7.16, 5.5))
    gs  = fig.add_gridspec(4, 2, hspace=0.1, wspace=0.15)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)])

    # share axes within each column
    for r in range(1, 4):
        axs[r, 0].sharey(axs[0, 0])
        axs[r, 1].sharey(axs[0, 1])

    for row, (model, color, label) in enumerate(row_configs):
        for col in range(2):
            ax = axs[row, col]

            y_true = norm_true if col == 0 else intg_true
            ax.plot(X_grid.squeeze(), y_true, color='black', linewidth=1.5,
                    label='Ground truth' if (row == 0 and col == 0) else '_')

            if col == 0:
                y_mean = np.array([
                    np.linalg.norm(model.get_mean(x.reshape(1, -1), [np.eye(yd) @ KT_half]))
                    for x in X_grid])
                bound  = np.array([
                    model.get_bound(x.reshape(1, -1), [np.eye(yd) @ KT_half])
                    for x in X_grid])
                ax.set_ylim(0,80)
            else:
                y_mean = np.array([
                    float(model.get_mean(x.reshape(1, -1), M_query))
                    for x in X_grid])
                bound  = np.array([
                    model.get_bound(x.reshape(1, -1), M_query)
                    for x in X_grid])
                ax.set_ylim(-3,3)

            ax.plot(X_grid.squeeze(), y_mean, color=color, linewidth=1.5,
                    label=label)
            ax.fill_between(X_grid.squeeze(),
                            np.maximum(y_mean - bound, 0) if col == 0 else y_mean - bound,
                            y_mean + bound, color=color, alpha=0.2)

            ax.set_xlim(0, 1)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(labelsize=12)
            ax.grid(True, linewidth=0.4)

            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('$x$', fontsize=12)

        axs[row, 0].set_ylabel(label, fontsize=12)

        axs[0, 0].set_title(r"$\left(\sum (f(x)(t_i))^2\right)^{1/2}$", fontsize=12)
        axs[0, 1].set_title(r'$M_{\mathrm{integral}}f(x)$', fontsize=12)

    # --- zoom insets on last row ---
    zoom_regions = {
        0: (0.4, 0.5, 48, 51),   # (x1, x2, y1, y2) for norm column
        1: (0.48, 0.52, -2, -1.85),   # for integral column
    }

    for col in range(2):
        ax = axs[3, col]
        x1, x2, y1, y2 = zoom_regions[col]

        ax_inset = ax.inset_axes(
            [0, 0.6, 0.3, 0.4],             # [left, bottom, width, height]
            xlim=(x1, x2), ylim=(y1, y2)
        )

        # re-plot ground truth
        y_true = norm_true if col == 0 else intg_true
        ax_inset.plot(X_grid.squeeze(), y_true, color='black', linewidth=1.0)

        # re-plot combined model only
        model, color, label = row_configs[-1]
        if col == 0:
            y_mean = np.array([
                np.linalg.norm(model.get_mean(x.reshape(1, -1), [np.eye(yd) @ KT_half]))
                for x in X_grid])
            bound  = np.array([
                model.get_bound(x.reshape(1, -1), [np.eye(yd) @ KT_half])
                for x in X_grid])
        else:
            y_mean = np.array([
                float(model.get_mean(x.reshape(1, -1), M_query))
                for x in X_grid])
            bound  = np.array([
                model.get_bound(x.reshape(1, -1), M_query)
                for x in X_grid])

        ax_inset.plot(X_grid.squeeze(), y_mean, color=color, linewidth=1.0)
        ax_inset.fill_between(X_grid.squeeze(),
                              np.maximum(y_mean - bound, 0) if col == 0 else y_mean - bound,
                              y_mean + bound, color=color, alpha=0.2)

        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)
        ax_inset.tick_params(labelsize=7)
        ax_inset.tick_params(labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(ax_inset, edgecolor='black', linewidth=0.8)

    legend_entries = [
        ('Ground truth',                           'black'),
        (r'$\mathcal{D}_{\mathrm{integral}}$',     'tab:green'),
        (r'$\mathcal{D}_{\mathrm{full}}$',         'tab:orange'),
        ('Combined',                               'tab:blue'),
        (r'$\mathcal{D}_{\mathrm{point}}$',        'tab:red'),
    ]

    proxy_handles = [Line2D([0], [0], color=c, linewidth=1.5) for _, c in legend_entries]
    proxy_labels  = [l for l, _ in legend_entries]

    fig.legend(proxy_handles, proxy_labels, fontsize=12, ncol=3,
               loc='upper center', bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(bottom=0.12)

    return fig

vvf, T, KT, KT_inv, KT_half, KT_integral,  t_node = get_true_vvf(f_true)

# fig, ax = comparison_norm_uncertainty(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral, n_meas=10)

plot_3d_prediction_and_error(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral, n_train=21, n_meas=5, meas_mode='point')
plt.savefig('1_input_RKHS_output_1.pdf', bbox_inches='tight', dpi=300)

# comparison_full_point(vvf, t_node, T, KT, KT_inv, KT_half, KT_integral, 
#                       x_queries=[0.5, 0.55, 0.25, 0.75],
#                       n_train_point=21, 
#                       n_train_full=3,
#                       n_meas=1)
# plt.savefig('1_input_RKHS_output_2.pdf', bbox_inches='tight', dpi=300)

# fig = comparison_integral_point_1(vvf, t_node, T, KT_inv, KT_half, KT_integral,x_query=0.5, x_meas=0.5,point_coord=[51],integral_coords=[51])
# plt.savefig('1_input_RKHS_output_3.pdf', bbox_inches='tight', dpi=300)

# fig = comparison_integral_point_2(
#     vvf, t_node, T, KT, KT_inv, KT_half, KT_integral,
#     x_query=0.5,
#     spec_base=[{'x': 0.5, 'mode': 'point', 'coords':[6,36,61,81,100]}],
#     spec_1=[{'x': 0.5, 'mode': 'integral'}],
#     spec_2=[{'x': 0.5, 'mode': 'integral'} for i in range(5)])
# plt.savefig('1_input_RKHS_output_4.pdf', bbox_inches='tight', dpi=300)

# fig = comparison_four_models(
#     vvf, t_node, T, KT, KT_inv, KT_half, KT_integral,
#     n_meas=5)
# plt.savefig('1_input_RKHS_output_5.pdf', bbox_inches='tight', dpi=300)

# also first test other basic functions
# currently, using a large reg. for KT works, but this also leads to a very large gamma, which is bad.
# think about what does it mean if KT is ill conditioned

plt.show()


'''
the non-uniform property of the uncertainty bound when there are only full measurement is cause by
two factors:
the first term: the inverse of Gram matrix, due to the numerical regulizar. The diagonal element is not fully equal. 
the thrid term (main contribution): the information uncertainty is almost equally distributed across t, but
MNI(t) - KRR(t) is not equally distributed, which follows the shape of the ground truth.
'''