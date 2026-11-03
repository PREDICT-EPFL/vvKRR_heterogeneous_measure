import numpy as np
import matplotlib.pyplot as plt
from regression.models import KRR, vvKRRPartial, vvKRR, vvKRRDirect
from regression.tools import Tools
from typing import Callable
from linear_system import simulate
from matplotlib.colors import PowerNorm
from matplotlib.colors import Normalize
import time

np.random.seed(0)
def f_true(X:np.ndarray):
    Y = []
    for i,v in enumerate(X):
        x_0 = np.array([v[0],0.5])
        u = np.array([[v[1]]])
        y,X = simulate(x_0,u,0.2,system = 'van_der_pol')
        Y.append(y)
    idx = np.linspace(0,19,20)
    T = Tools.Kernel.RBF(idx.reshape(-1,1),length_scale = 2)
    return Y, T

def get_true_vvf(f_true:Callable):
    X_train = Tools.others.make_grid(np.array([[-1,1],[0,1]]),10)
    Y_train , T = f_true(X_train)
    M_list = [np.eye(20) for i in range(np.shape(X_train)[0])]

    vvf = vvKRRPartial(X_train, M_list, Y_train, Tools.Kernel.RBF, T, length_scale = np.array([0.5,0.2]), gamma = 5, delta_ub = 0.1, bound = np.array([[-1,1],[0,1]]), lambda_ = 0.01, if_delta= False)
    return vvf, T

def sample_unit_ball_n(N, n_samples,rng):
    z = rng.normal(size=(n_samples, N))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / norms
    
    r = rng.random(n_samples) ** (1.0 / N)
    r = r.reshape(-1, 1)
    
    return r * u

def generate_training_data(vvf, delta_ub=0.01, n_meas=15):
    rng = np.random.default_rng(seed=0)

    yd = 20

    x1 = np.linspace(-1, 1, 11)
    x2 = np.linspace(0, 1, 11)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_train = np.column_stack([xx1.ravel(), xx2.ravel()])  # (121, 2)
    num_sample = X_train.shape[0]

    M_list = []
    Y_train = []
    meas_type = []
    Y_train_full = np.full((num_sample, yd), np.nan)

    scalar_data = {j: {'X': [], 'Y': []} for j in range(yd)}

    for i, v in enumerate(X_train):
        x = np.atleast_2d(v)
        fx = vvf.get_mean(x, [np.eye(yd)])  # shape (yd,)

        coords = np.sort(rng.choice(yd, size=n_meas, replace=False))
        Mi = np.eye(yd)[coords]  # shape (n_meas, yd)
        M_list.append(Mi)
        meas_type.append(tuple(coords))

        noise_i = sample_unit_ball_n(n_meas, 1, rng).squeeze() * delta_ub
        yi = Mi @ fx + noise_i  # shape (n_meas,)
        Y_train.append(yi)

        Y_train_full[i, coords] = yi

        for k, coord in enumerate(coords):
            scalar_data[coord]['X'].append(v)
            scalar_data[coord]['Y'].append(yi[k])

    for j in range(yd):
        scalar_data[j]['X'] = np.array(scalar_data[j]['X'])
        scalar_data[j]['Y'] = np.array(scalar_data[j]['Y'])

    return X_train, Y_train, Y_train_full, M_list, meas_type, scalar_data

def plot_output_maps(vvf, X_train, scalar_data, yd=20, resolution=30):
    """
    Plot the predicted map and training samples for each output dimension.
    
    vvf       : fitted vvKRR model
    X_train   : (121, 2) training inputs
    scalar_data: dict from generate_training_data
    yd        : output dimension
    resolution: grid resolution for contour
    """
    # Evaluation grid
    x1 = np.linspace(-1, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])  # (res^2, 2)

    # Evaluate model on grid
    F_grid = np.array([
        vvf.get_mean(np.atleast_2d(x), [np.eye(yd)]) for x in X_grid
    ])  # (res^2, yd)

    # Layout: 4 rows x 5 cols for 20 dims
    n_cols = 5
    n_rows = int(np.ceil(yd / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 3.5, n_rows * 3),
                            constrained_layout=True)
    axs = axs.ravel()

    for j in range(yd):
        ax = axs[j]
        Z = F_grid[:, j].reshape(resolution, resolution)

        # Contour plot of predicted map
        cf = ax.contourf(xx1, xx2, Z, levels=20, cmap='viridis')
        ax.contour(xx1, xx2, Z, levels=10, colors='white', linewidths=0.4, alpha=0.5)
        fig.colorbar(cf, ax=ax, shrink=0.85)

        # Overlay samples where dimension j was measured
        Xj = scalar_data[j]['X']  # (n_j, 2)
        Yj = scalar_data[j]['Y']  # (n_j,)
        if len(Xj) > 0:
            sc = ax.scatter(Xj[:, 0], Xj[:, 1],
                            c=Yj, cmap='viridis',
                            edgecolors='white', linewidths=0.5,
                            s=40, zorder=5)

        ax.set_title(f'dim {j}', fontsize=11)
        ax.set_xlabel('$x_1$', fontsize=9)
        ax.set_ylabel('$x_2$', fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(yd, len(axs)):
        axs[j].set_visible(False)

    return fig, axs


def plot_input_and_trajectories(scalar_data, vvf, test_inputs, yd=20, tol=1e-8):
    """
    test_inputs : list of 3 arrays, each shape (1, 2)
    Left  : 2D input space (x-axis: x2, y-axis: x1)
    Right : ground truth trajectories at test_inputs + samples at nearest training point
    """
    dims, x1s, x2s, vals = [], [], [], []
    for j in range(yd):
        Xj = scalar_data[j]['X']
        Yj = scalar_data[j]['Y']
        for k in range(len(Xj)):
            dims.append(j)
            x1s.append(Xj[k, 0])
            x2s.append(Xj[k, 1])
            vals.append(Yj[k])
    dims = np.array(dims)
    x1s  = np.array(x1s)
    x2s  = np.array(x2s)
    vals = np.array(vals)

    unique_inputs = np.unique(np.column_stack([x1s, x2s]), axis=0)

    dim_axis = np.arange(yd)
    colors   = ['tab:blue', 'tab:orange', 'tab:green']
    markers  = ['o', 'o', 'o']

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3))
    fig.subplots_adjust(wspace=0.4)
    ax_in, ax_traj = axs

    # ---- Left: input space ----
    ax_in.scatter(unique_inputs[:, 1], unique_inputs[:, 0],
                  color='tab:grey', s=20, alpha=0.4, label='Training inputs')

    for idx, x_in in enumerate(test_inputs):
        dists = np.linalg.norm(unique_inputs - x_in[0], axis=1)
        nearest = unique_inputs[np.argmin(dists)]

        ax_in.scatter(nearest[1], nearest[0],
                      color=colors[idx], marker=markers[idx],
                      s=50, zorder=6, edgecolors='black', linewidths=1.2)

    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(-1, 1)
    # ax_in.set_aspect(0.8)
    ax_in.set_xlabel('u', fontsize=12)
    ax_in.set_ylabel('$v_1(0)$', fontsize=12)
    # ax_in.yaxis.set_label_coords(-0.28, 0.5)
    # ax_in.set_title('Input space', fontsize=12)
    # ax_in.legend(fontsize=12, loc='upper right')
    # ax_in.grid(True, alpha=0.3)
    ax_in.yaxis.set_major_locator(plt.MaxNLocator(6))

    # ---- Right: trajectories + samples ----
    all_dims, all_vals = [], []

    for idx, x_in in enumerate(test_inputs):
        dists = np.linalg.norm(unique_inputs - x_in[0], axis=1)
        nearest = unique_inputs[np.argmin(dists)]
        mask = (np.abs(x1s - nearest[0]) < tol) & (np.abs(x2s - nearest[1]) < tol)

        gt = vvf.get_mean(x_in, [np.eye(yd)])
        ax_traj.plot(dim_axis, gt,
                     color=colors[idx], linewidth=1.8, alpha=0.8, markersize=4,
                     label=f'$v_1(0):{x_in[0,0]},u:{x_in[0,1]}$')

        all_dims.append(dims[mask])
        all_vals.append(vals[mask])

    ax_traj.scatter(np.concatenate(all_dims), np.concatenate(all_vals),
                    color='tab:red', s=40, alpha=0.9,
                    edgecolors='white', linewidths=0.5, zorder=10)

    ax_traj.set_xlabel('Time step', fontsize=12)
    ax_traj.set_ylabel('f(x)', fontsize=12)
    # ax_traj.yaxis.set_label_coords(-0.16, 0.5)
    ax_traj.set_xticks(range(0, yd, 2))
    ax_traj.set_xlim(0, yd - 1)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=8, loc='upper right',
                handlelength=1.5)
    ax_traj.yaxis.set_major_locator(plt.MaxNLocator(6))

    return fig, axs

def plot_trajectory_comparison(vvf, T, f_true, test_input, n_meas_list=[5, 10, 15],
                                delta_dict=None, yd=20):
    X_grid = np.arange(yd)
    vvKRR_true = vvf.get_mean(test_input, [np.eye(yd)])
    M_list_bound = [np.eye(1, yd, k=i) for i in range(yd)]

    fig, axs = plt.subplots(len(n_meas_list), 1,
                            figsize=(7.16, 6),
                            sharex=True)
    fig.subplots_adjust(hspace=0.15, bottom=0.18)

    for i, (ax, n_meas) in enumerate(zip(axs, n_meas_list)):
        X_train, Y_train, _, M_list, _, scalar_data = generate_training_data(vvf, n_meas=n_meas)

        vvKRR_model = vvKRRPartial(X_train, M_list, Y_train,
                                   Tools.Kernel.RBF, T=T,
                                   length_scale=np.array([0.5, 0.2]),
                                   gamma=vvf.norm, delta_ub=0.01,
                                   bound=np.array([[-1,1],[0,1]]),
                                   lambda_=0.01, if_delta=False,
                                   Delta=delta_dict[n_meas])

        scalarKRR_model_list = [KRR(scalar_data[j]['X'], scalar_data[j]['Y'],
                                    Tools.Kernel.RBF,
                                    length_scale=np.array([0.5, 0.2]),
                                    gamma=vvf.norm, delta_ub=0.01,
                                    bound=np.array([[-1,1],[0,1]]),
                                    lambda_=0.01) for j in range(yd)]

        vvKRR_test   = vvKRR_model.get_mean(test_input, [np.eye(yd)])
        vvKRR_bound  = np.array([vvKRR_model.get_bound(test_input, [m]) for m in M_list_bound])
        scalarKRR_test  = np.array([scalarKRR_model_list[j].get_mean(test_input) for j in range(yd)])
        scalarKRR_bound = np.array([scalarKRR_model_list[j].get_bound(test_input) for j in range(yd)])

        ax.fill_between(X_grid,
                        scalarKRR_test - scalarKRR_bound,
                        scalarKRR_test + scalarKRR_bound,
                        color='tab:orange', alpha=0.3)
        ax.fill_between(X_grid,
                        vvKRR_test - vvKRR_bound,
                        vvKRR_test + vvKRR_bound,
                        color='tab:blue', alpha=0.3)

        ax.plot(X_grid, vvKRR_true,     color='black',    linewidth=2,
                markersize=8, label='Ground truth')
        ax.plot(X_grid, vvKRR_test,     color='tab:blue', linewidth=2,
                markersize=5, label='Vector-valued model', zorder=10)
        ax.plot(X_grid, scalarKRR_test, color='tab:orange', linewidth=2,
                markersize=5, label='Scalar-valued model')

        ax.text(0.98, 0.95, f'{n_meas} measurements',
                transform=ax.transAxes,
                fontsize=12, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.8', alpha=0.8))
        
        ax.set_ylabel('f(x)',fontsize = 12)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(0, yd - 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    axs[-1].set_xlabel('Time step', fontsize=12)
    axs[-1].set_xticks(X_grid)
    # fig.suptitle(f'Trajectory at $x = {test_input.ravel().tolist()}$', fontsize=13)

    # Shared legend below all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=3,
               fontsize=12, framealpha=0.9,
               bbox_to_anchor=(0.5, 0))

    return fig, axs

def plot_uncertainty_comparison(vvf, T, dims, n_meas=5,
                                 delta_dict=None, yd=20, resolution=30):
    x1 = np.linspace(-1, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    X_train, Y_train, _, M_list, _, sd = generate_training_data(vvf, n_meas=n_meas)

    vvKRR_model = vvKRRPartial(X_train, M_list, Y_train,
                               Tools.Kernel.RBF, T=T,
                               length_scale=np.array([0.5, 0.2]),
                               gamma=vvf.norm, delta_ub=0.01,
                               bound=np.array([[-1,1],[0,1]]),
                               lambda_=0.01, if_delta=False,
                               Delta=delta_dict[n_meas])

    scalarKRR_model_list = [KRR(sd[i]['X'], sd[i]['Y'],
                                Tools.Kernel.RBF,
                                length_scale=np.array([0.5, 0.2]),
                                gamma=vvf.norm, delta_ub=0.01,
                                bound=np.array([[-1,1],[0,1]]),
                                lambda_=0.01) for i in range(yd)]

    vv_bounds_list     = []
    scalar_bounds_list = []

    for dim in dims:
        M_dim = [np.eye(1, yd, k=dim)]
        vv_bounds = np.array([vvKRR_model.get_bound(np.atleast_2d(x), M_dim)
                               for x in X_grid])
        scalar_bounds = np.array([scalarKRR_model_list[dim].get_bound(np.atleast_2d(x))
                                   for x in X_grid])
        vv_bounds_list.append(vv_bounds)
        scalar_bounds_list.append(scalar_bounds)

    all_vals = np.concatenate(vv_bounds_list + scalar_bounds_list)
    vmin, vmax = all_vals.min(), all_vals.max()

    # 2 rows x 2 cols: row 0 = vvKRR, row 1 = Scalar
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.16, 7.16),
                            constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    row_labels = ['Vector-valued model', 'Scalar-valued model']
    panels = [
        [(vv_bounds_list[0],     dims[0]), (vv_bounds_list[1],     dims[1])],
        [(scalar_bounds_list[0], dims[0]), (scalar_bounds_list[1], dims[1])],
    ]

    for row in range(2):
        for col in range(2):
            ax  = axs[row, col]
            bounds, dim = panels[row][col]
            Xj  = sd[dim]['X']
            Z   = bounds.reshape(resolution, resolution)

            cf = ax.contourf(xx2, xx1, Z, levels=20, cmap='viridis',
                             vmin=vmin, vmax=vmax, antialiased=True)

            if len(Xj) > 0:
                ax.scatter(Xj[:, 1], Xj[:, 0],
                           color='tab:red', edgecolors='white',
                           linewidths=0.5, s=25, zorder=5)

            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            # ax.set_aspect('equal')
            ax.tick_params(labelsize=12)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            if row == 1:
                ax.set_xlabel('$u$', fontsize=12)
            if col == 0:
                ax.set_ylabel('$v_1(0)$', fontsize=12)
                ax.yaxis.set_label_coords(-0.15, 0.5)
            if row == 0:
                ax.set_title(f'Time step {dim}', fontsize=12)
            if col == 0:
                ax.text(-0.25, 0.5, row_labels[row], transform=ax.transAxes,
                        fontsize=12, va='center', ha='center', rotation=90)

    cb = fig.colorbar(cf, ax=axs, shrink=0.5, pad=0.02, location='bottom')
    cb.ax.tick_params(labelsize=12)
    cb.ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    return fig, axs

def plot_norm_error(vvf, T, n_meas_list=[5, 10, 15],
                          delta_dict=None, yd=20, resolution=30,zlim = 1.6):
    '''
    plot the prodiction error in norm
    '''
    x1 = np.linspace(-1, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    gt_grid = np.array([vvf.get_mean(np.atleast_2d(x), [np.eye(yd)]) for x in X_grid])

    vv_error_list     = []
    scalar_error_list = []
    vv_bound_list     = []
    scalar_bound_list = []

    for n_meas in n_meas_list:
        print(n_meas)
        X_train, Y_train, _, M_list, _, sd = generate_training_data(vvf, n_meas=n_meas)

        vvKRR_model = vvKRRPartial(X_train, M_list, Y_train,
                                   Tools.Kernel.RBF, T=T,
                                   length_scale=np.array([0.5, 0.2]),
                                   gamma=vvf.norm, delta_ub=0.01,
                                   bound=np.array([[-1,1],[0,1]]),
                                   lambda_=0.01, if_delta=False,
                                   Delta=delta_dict[n_meas])

        scalarKRR_model_list = [KRR(sd[i]['X'], sd[i]['Y'],
                                    Tools.Kernel.RBF,
                                    length_scale=np.array([0.5, 0.2]),
                                    gamma=vvf.norm, delta_ub=0.01,
                                    bound=np.array([[-1,1],[0,1]]),
                                    lambda_=0.01) for i in range(yd)]

        vv_mu = np.array([vvKRR_model.get_mean(np.atleast_2d(x), [np.eye(yd)]) for x in X_grid])
        sc_mu = np.array([np.array([scalarKRR_model_list[j].get_mean(np.atleast_2d(x))
                          for j in range(yd)]) for x in X_grid])

        M_full = [np.eye(yd)]
        
        vv_bound = np.array([vvKRR_model.get_bound(np.atleast_2d(x), M_full)
                            for x in X_grid])
        sc_bound = np.array([
            np.sqrt(np.sum(np.array([scalarKRR_model_list[j].get_bound(np.atleast_2d(X_grid[i]))
                            for j in range(yd)]) ** 2))
            for i in range(len(X_grid))
        ])

        vv_error_list.append(np.linalg.norm(vv_mu - gt_grid, axis=1))
        scalar_error_list.append(np.linalg.norm(sc_mu - gt_grid, axis=1))
        vv_bound_list.append(vv_bound)
        scalar_bound_list.append(sc_bound)

    all_vals = np.concatenate(vv_error_list + scalar_error_list)
    vmin_val = max(all_vals.min(), 1e-6)
    vmax_val = all_vals.max()
    norm = PowerNorm(gamma=0.3, vmin=vmin_val, vmax=vmax_val)
    cmap = plt.cm.viridis

    fig, axs = plt.subplots(2, 3, figsize=(7.16, 5),
                            subplot_kw={'projection': '3d'},
                            constrained_layout=True)

    row_labels = ['Vector-valued model', 'Scalar-valued model']
    panels = [
        [(vv_error_list[i], vv_bound_list[i],     n_meas_list[i]) for i in range(3)],
        [(scalar_error_list[i], scalar_bound_list[i], n_meas_list[i]) for i in range(3)],
    ]

    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            errors, bounds, n_meas = panels[row][col]
            Z       = errors.reshape(resolution, resolution)
            Z_bound = bounds.reshape(resolution, resolution)

            Z_clipped       = np.clip(Z,       vmin_val, zlim)
            Z_bound_clipped = np.clip(Z_bound, vmin_val, zlim)

            # Error surface
            facecolors = cmap(norm(Z_clipped))
            ax.plot_surface(xx2, xx1, Z_clipped, facecolors=facecolors,
                            rstride=1, cstride=1, antialiased=False,
                            shade=False, zorder=2)

            # Bound surface — transparent red
            ax.plot_surface(xx2, xx1, Z_bound_clipped,
                            color='tab:grey', alpha=0.4,
                            rstride=1, cstride=1, antialiased=False,
                            shade=False, zorder=1)

            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(vmin_val, zlim)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('$u$', fontsize=9, labelpad=2)
            ax.set_ylabel('$v_1(0)$', fontsize=9, labelpad=2)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.zaxis.set_major_locator(plt.MaxNLocator(4))

            if row == 0:
                ax.set_title(f'{n_meas} measurements', fontsize=10)
            if col == 0:
                ax.text2D(-0.15, 0.5, row_labels[row], transform=ax.transAxes,
                          fontsize=10, va='center', ha='center', rotation=90)
            if col != 2:
                ax.zaxis.set_ticklabels([])
                ax.set_zlabel('')
            else:
                pass
            
    return fig, axs

def plot_norm(vvf, T, delta_dict, yd=20, resolution=30, zlim=1.6):
    from matplotlib.colors import Normalize

    x1 = np.linspace(-1, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    M_full = [np.eye(yd)]

    # 15 measurements — scalar KRR and vvKRRPartial share the same training data
    X_train, Y_train, _, M_list, _, sd = generate_training_data(vvf, n_meas=15)

    scalarKRR_15 = [KRR(sd[i]['X'], sd[i]['Y'],
                        Tools.Kernel.RBF,
                        length_scale=np.array([0.5, 0.2]),
                        gamma=vvf.norm, delta_ub=0.01,
                        bound=np.array([[-1,1],[0,1]]),
                        lambda_=0.01) for i in range(yd)]
    sc_mu_15 = np.array([np.array([scalarKRR_15[j].get_mean(np.atleast_2d(x)) for j in range(yd)])
                         for x in X_grid])
    sc_b_15  = np.array([np.sqrt(np.sum(np.array([scalarKRR_15[j].get_bound(np.atleast_2d(x))
                                                   for j in range(yd)]) ** 2))
                         for x in X_grid])
    sc_norm_15 = np.linalg.norm(sc_mu_15, axis=1)

    vvKRR_15 = vvKRRPartial(X_train, M_list, Y_train,
                             Tools.Kernel.RBF, T=T,
                             length_scale=np.array([0.5, 0.2]),
                             gamma=vvf.norm, delta_ub=0.01,
                             bound=np.array([[-1,1],[0,1]]),
                             lambda_=0.01, if_delta=False,
                             Delta=delta_dict[15])
    mu_15      = np.array([vvKRR_15.get_mean(np.atleast_2d(x), M_full) for x in X_grid])
    b_15       = np.array([vvKRR_15.get_bound(np.atleast_2d(x), M_full) for x in X_grid])
    mu_norm_15 = np.linalg.norm(mu_15, axis=1)

    # Full measurement — vvKRR
    X_train_full, _, Y_train_full, _, _, _ = generate_training_data(vvf, n_meas=yd)
    Y_train_full = np.array(Y_train_full)
    vvKRR_full = vvKRR(X_train_full, Y_train_full,
                       Tools.Kernel.RBF, T=T,
                       length_scale=np.array([0.5, 0.2]),
                       gamma=vvf.norm, delta_ub=0.01,
                       bound=np.array([[-1,1],[0,1]]),
                       lambda_=0.01)
    mu_full      = np.array([vvKRR_full.get_mean(np.atleast_2d(x)) for x in X_grid])
    b_full       = np.array([vvKRR_full.get_bound(np.atleast_2d(x)) for x in X_grid])
    mu_norm_full = np.linalg.norm(mu_full, axis=1)

    # Global color range
    all_vals = np.concatenate([sc_norm_15, mu_norm_15, mu_norm_full])
    vmin_val = all_vals.min()
    vmax_val = all_vals.max()
    norm_color = Normalize(vmin=vmin_val, vmax=vmax_val)
    cmap = plt.cm.viridis

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5),
                            subplot_kw={'projection': '3d'},
                            constrained_layout=True)

    panels = [
        (sc_norm_15,   sc_b_15, 'Scalar model'),
        (mu_norm_15,   b_15,    'Vector model'),
        (mu_norm_full, b_full,  'Vector model (full meas.)'),
    ]

    for i, (ax, (mu_norm, bound, title)) in enumerate(zip(axs, panels)):
        Z         = mu_norm.reshape(resolution, resolution)
        Z_clipped = np.clip(Z, vmin_val, zlim)

        facecolors = cmap(norm_color(Z_clipped))
        ax.plot_surface(xx2, xx1, Z_clipped, facecolors=facecolors,
                        rstride=1, cstride=1, antialiased=False,
                        shade=False, zorder=2)

        if bound is not None:
            Z_ub = np.clip((mu_norm + bound).reshape(resolution, resolution), vmin_val, zlim)
            Z_lb = np.clip((mu_norm - bound).reshape(resolution, resolution), 0, zlim)
            ax.plot_surface(xx2, xx1, Z_ub,
                            color='tab:grey', alpha=0.4,
                            rstride=1, cstride=1, antialiased=False,
                            shade=False, zorder=3)
            ax.plot_surface(xx2, xx1, Z_lb,
                            color='tab:grey', alpha=0.4,
                            rstride=1, cstride=1, antialiased=False,
                            shade=False, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, zlim)
        ax.tick_params(labelsize=11)
        ax.set_xlabel('$u$',   fontsize=12, labelpad=2)
        ax.set_ylabel('$v_1(0)$', fontsize=11, labelpad=2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_title(title, fontsize=12)

        if i != 2:
            ax.zaxis.set_ticklabels([])
            ax.set_zlabel('')

    return fig, axs

def plot_norm_region(vvKRR_model, scalarKRR_model_list, vvf, c, yd=20, resolution=30):
    """
    Plot the region where the output norm is <= c.
    vvKRR   : ||mu(x)|| + bound(x, I) <= c
    scalar  : sqrt(sum_i (|mu_i(x)| + bound_i(x))^2) <= c
    truth   : ||f(x)|| <= c
    """
    x1 = np.linspace(-1, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    M_full = [np.eye(yd)]

    vv_vals      = np.zeros(len(X_grid))
    scalar_vals  = np.zeros(len(X_grid))
    truth_vals   = np.zeros(len(X_grid))

    for i, x in enumerate(X_grid):
        xq = np.atleast_2d(x)

        # vvKRR: ||mu|| + bound(I)
        mu_vv  = vvKRR_model.get_mean(xq, M_full)
        # b_vv   = vvKRR_model.get_bound(xq, M_full)
        # vv_vals[i] = np.linalg.norm(mu_vv) + b_vv
        vv_vals[i] = np.linalg.norm(mu_vv)
        

        # Scalar: sqrt(sum_i (|mu_i| + bound_i)^2)
        mu_s = np.array([scalarKRR_model_list[j].get_mean(xq) for j in range(yd)])
        # b_s  = np.array([scalarKRR_model_list[j].get_bound(xq) for j in range(yd)])
        # scalar_vals[i] = np.sqrt(np.sum((np.abs(mu_s) + b_s) ** 2))
        scalar_vals[i] = np.sqrt(np.sum((np.abs(mu_s)) ** 2))

        # Ground truth
        ft = vvf.get_mean(xq, M_full)
        truth_vals[i] = np.linalg.norm(ft)

    VV     = vv_vals.reshape(resolution, resolution)
    SC     = scalar_vals.reshape(resolution, resolution)
    GT     = truth_vals.reshape(resolution, resolution)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)

    titles = [f'vvKRR bound ($c={c}$)',
              f'Scalar KRR bound ($c={c}$)',
              f'Ground truth ($c={c}$)']

    for ax, Z, title in zip(axs, [VV, SC, GT], titles):
        # Binary region
        ax.contourf(xx1, xx2, Z, levels=[0, c], colors=['tab:blue'], alpha=0.4)
        ax.contourf(xx1, xx2, Z, levels=[0, c], colors=['tab:blue'], alpha=0.4)
        if Z.max() > c:
            ax.contourf(xx1, xx2, Z, levels=[c, Z.max() + 1], colors=['tab:red'], alpha=0.2)
        ax.contour(xx1, xx2, Z, levels=[c], colors='black', linewidths=1.5)
        ax.contour(xx1, xx2, Z, levels=[c], colors='black', linewidths=1.5)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:blue', alpha=0.4, label=r'$\leq c$'),
                       Patch(facecolor='tab:red',  alpha=0.4, label=r'$> c$')]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(r'Region where $\|f(x)\| \leq c$', fontsize=13)
    return fig, axs

def plot_norm_uncertainty_slice(vvKRR_model, vvf, x1_fixed, x2_fixed, yd=20, resolution=50):
    """
    Two subplots:
    Left : fix x1, vary x2 — plot ||mu|| +- bound and ||f|| vs x2
    Right: fix x2, vary x1 — plot ||mu|| +- bound and ||f|| vs x1
    """
    M_full  = [np.eye(yd)]
    x2_grid = np.linspace(0, 1,  resolution)
    x1_grid = np.linspace(-1, 1, resolution)

    def compute(grid, fixed, fix_x1):
        gt_norm, mu_norm, bound = [], [], []
        for val in grid:
            x_in = np.array([[fixed, val]]) if fix_x1 else np.array([[val, fixed]])
            gt_norm.append(np.linalg.norm(vvf.get_mean(x_in, M_full)))
            mu_hat = vvKRR_model.get_mean(x_in, M_full)
            mu_norm.append(np.linalg.norm(mu_hat))
            bound.append(vvKRR_model.get_bound(x_in, M_full))
        return np.array(gt_norm), np.array(mu_norm), np.array(bound)

    gt1, mu1, b1 = compute(x2_grid, x1_fixed,  fix_x1=True)
    gt2, mu2, b2 = compute(x1_grid, x2_fixed, fix_x1=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for ax, grid, gt, mu, b, xlabel, title in zip(
        axs,
        [x2_grid, x1_grid],
        [gt1, gt2],
        [mu1, mu2],
        [b1, b2],
        ['$x_2$', '$x_1$'],
        [f'Fixed $x_1={x1_fixed}$, vary $x_2$',
         f'Fixed $x_2={x2_fixed}$, vary $x_1$'],
    ):
        ax.fill_between(grid, mu - b, mu + b,
                        color='tab:blue', alpha=0.3,
                        label=r'$\|\mu(x)\| \pm \epsilon(x)$')
        ax.plot(grid, gt, color='black',    linewidth=2, label=r'$\|f(x)\|$')
        ax.plot(grid, mu, color='tab:blue', linewidth=2, label=r'$\|\mu(x)\|$')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'$\|\cdot\|$', fontsize=12)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    return fig, axs

def check_vvf_alignment(vvf, resolution=20):
    '''
    check if the vvf approximate the ground truth function well
    '''
    x1_grid = np.linspace(-1, 1, resolution)
    x2_grid = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    errors = []
    y_rkhs_list = []
    y_true_list = []

    for v in X_grid:
        x_in  = np.atleast_2d(v)
        y_rkhs = vvf.get_mean(x_in, [np.eye(20)])

        x_0 = np.array([v[0], 0.5])
        u   = np.array([[v[1]]])
        y_true, _ = simulate(x_0, u, 0.2, system='van_der_pol')

        errors.append(np.linalg.norm(y_rkhs - y_true))
        y_rkhs_list.append(y_rkhs)
        y_true_list.append(y_true)

    errors     = np.array(errors)
    errors_2d  = errors.reshape(resolution, resolution)
    dim_axis   = np.arange(20)

    # Top-3 worst cases
    top_k = 3
    top_idx = np.argsort(errors)[-top_k:][::-1]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # ---- Left: contour of error ----
    cf = axs[0].contourf(xx2, xx1, errors_2d, levels=20, cmap='viridis', antialiased=False)
    for idx in top_idx:
        axs[0].scatter(X_grid[idx, 1], X_grid[idx, 0],
                       color='tab:red', s=60, zorder=5, edgecolors='white', linewidths=0.6)
    fig.colorbar(cf, ax=axs[0], label=r'$\|vvf(x) - f(x)\|$')
    axs[0].set_xlabel('$x_2$', fontsize=12)
    axs[0].set_ylabel('$x_1$', fontsize=12)
    axs[0].set_title('Pointwise error', fontsize=12)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(-1, 1)

    # ---- Middle: histogram ----
    axs[1].hist(errors.ravel(), bins=30, color='tab:blue', edgecolor='white')
    axs[1].set_xlabel(r'$\|vvf(x) - f(x)\|$', fontsize=12)
    axs[1].set_ylabel('Count', fontsize=12)
    axs[1].set_title(f'Max={errors.max():.4f}, Mean={errors.mean():.4f}', fontsize=11)
    axs[1].grid(True, alpha=0.3)

    # ---- Right: trajectories at worst cases ----
    colors = ['tab:red', 'tab:orange', 'tab:green']
    for rank, idx in enumerate(top_idx):
        v     = X_grid[idx]
        color = colors[rank]
        label = f'$x=({v[0]:.2f},{v[1]:.2f})$, err={errors[idx]:.3f}'
        axs[2].plot(dim_axis, y_true_list[idx],  color=color, linewidth=2,
                    linestyle='-',  label=f'true {label}')
        axs[2].plot(dim_axis, y_rkhs_list[idx],  color=color, linewidth=2,
                    linestyle='--', label=f'vvf {label}')

    axs[2].set_xlabel('Output dimension', fontsize=12)
    axs[2].set_ylabel('Value', fontsize=12)
    axs[2].set_title('Trajectories at max error inputs', fontsize=12)
    axs[2].set_xlim(0, 19)
    axs[2].set_xticks(range(0, 20, 2))
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=7, loc='upper right')

    print(f'Max error  : {errors.max():.6f}')
    print(f'Mean error : {errors.mean():.6f}')
    for rank, idx in enumerate(top_idx):
        print(f'  Top-{rank+1}: x={X_grid[idx]}, error={errors[idx]:.6f}')

    return fig, axs, errors

vvf, T = get_true_vvf(f_true)
n_meas = 5
delta_dict = {5:11.481239907610599,
              10:35.14461692357056,
              15:56.07458477936917,
              20:56}

X_train, Y_train, Y_train_full, M_list, meas_type, scalar_data = generate_training_data(vvf,n_meas = n_meas)

vvKRR_model = vvKRRPartial(X_train, 
                    M_list,
                    Y_train,
                    Tools.Kernel.RBF, 
                    T = T,
                    length_scale = np.array([0.5,0.2]),
                    gamma = vvf.norm, 
                    delta_ub = 0.01,
                    bound = np.array([[-1,1],[0,1]]), 
                    lambda_ = 0.01,
                    if_delta=False,
                    Delta = delta_dict[n_meas])

scalarKRR_model_list = [KRR(scalar_data[i]['X'], 
                        scalar_data[i]['Y'], 
                        Tools.Kernel.RBF, 
                        length_scale = np.array([0.5,0.2]), 
                        gamma = vvf.norm,
                        delta_ub = 0.01, 
                        bound = np.array([[-1,1],[0,1]]), 
                        lambda_ = 0.01) for i in range(20)]

# test_inputs = [np.array([[-0.8, 0.2]]),
#                np.array([[ -0.2, 0.5]]),
#                np.array([[ 0, 0.8]])]
# fig, axs = plot_input_and_trajectories(scalar_data, vvf, test_inputs)
# plt.savefig('2_input_20_output_1.pdf', bbox_inches='tight', dpi=300)

# fig, axs = plot_trajectory_comparison(vvf, T, f_true,test_input=np.array([[-0.5, 0.5]]),n_meas_list=[5, 10, 15],delta_dict=delta_dict)
# plt.savefig('2_input_20_output_2.pdf', bbox_inches='tight', dpi=300)

fig, axs = plot_uncertainty_comparison(vvf, T, dims=[5, 6], n_meas=5, delta_dict=delta_dict)
plt.savefig('2_input_20_output_3.png', bbox_inches='tight', dpi=300)

# fig, axs = plot_norm_error(vvf, T, n_meas_list=[5, 10, 15], delta_dict=delta_dict,zlim=1)
# plt.savefig('2_input_20_output_4.pdf', bbox_inches='tight', dpi=300)

fig, axs = plot_norm(vvf, T, delta_dict, zlim = 3)
plt.savefig('2_input_20_output_5.pdf', bbox_inches='tight', dpi=300)

# fig, axs = plot_norm_uncertainty_slice(vvKRR_model, vvf, x1_fixed=-0.5, x2_fixed=0.5)

# fig, axs, errors = check_vvf_alignment(vvf)

# plt.show()
