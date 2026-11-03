# vvRKHS

Vector-valued kernel ridge regression (vv-KRR) with **partial linear measurements**
and deterministic uncertainty bounds.

The model learns a vector-valued function `f: X -> Y` from data where each sample
only reveals a linear projection of the output, `y_i = M_i f(x_i) + noise`
(for example a single output component, the sum of components, or point and
integral evaluations of a function-valued output). The scripts compare the
resulting vector-valued model and its error bounds with independent scalar KRR
models fitted to each output.

## Repository layout

```
vvRKHS/
├── regression/                         # Model library
│   ├── basemodel.py                    #   Abstract model interface
│   ├── models.py                       #   KRR / vv-KRR model classes
│   └── tools.py                        #   Kernels (NumPy and CasADi) and grid utilities
├── linear_system.py                    # Dynamical-system simulator (rotation, Van der Pol, Duffing)
├── vvRKHS_2d_partial.py                # 1-D input, 2-D output, separable kernel
├── vvRKHS_2d_partial_nonseparable.py   # 1-D input, 2-D output, non-separable kernel
├── vvRKHS_nd_partial.py                # 2-D input, 20-D output (system trajectories)
└── vvRKHS_regression_gp.py             # 1-D input, function-valued (RKHS) output
```

The `regression` package is a trimmed copy of
[Wenbin24/regression](https://github.com/Wenbin24/regression). It keeps only the
classes used here, with their code unchanged, so this repository runs on its own
without Gurobi or other optional dependencies.

## Models

All models live in `regression/models.py` and share the `fit`, `get_mean`, and
`get_bound` interface. Bounds are computed by solving a small optimisation problem
with CasADi/IPOPT, using the RKHS norm bound `gamma` and the per-sample noise
bound `delta_ub`.

| Class | Description |
|---|---|
| `KRR` | Scalar kernel ridge regression with uncertainty bounds |
| `KRRNonseparableKernel` | Scalar KRR with a weighted sum of RBF kernels |
| `vvKRR` | vv-KRR with separable kernel `k(x, x') T`, fully observed outputs |
| `vvKRRDirect` | vv-KRR on the full Kronecker Gram matrix `G ⊗ T` |
| `vvKRRFunctionOutput` | vv-KRR whose outputs are coefficient vectors of a function basis |
| `vvKRRPartial` | vv-KRR with partial linear measurements `M_i`, separable kernel |
| `vvKRRPartialNonseparableKernel` | vv-KRR with partial measurements and a non-separable kernel |

Minimal example:

```python
import numpy as np
from regression.models import vvKRRPartial
from regression.tools import Tools

X = np.random.rand(20, 1)                       # inputs, shape (n, d)
T = np.eye(2)                                   # output covariance operator
M = [np.array([[1.0, 0.0]])] * 20               # each sample observes only f_1
Y = [np.sin(2 * np.pi * x) for x in X]          # measurements M_i f(x_i)

model = vvKRRPartial(X, M, Y, Tools.Kernel.RBF, T=T, length_scale=0.1,
                     gamma=5, delta_ub=0.1, bound=np.array([[0, 1]]), lambda_=0.01)

x = np.array([[0.3]])
mean  = model.get_mean(x, [np.eye(2)])              # full output prediction
bound = model.get_bound(x, [np.array([[0, 1]])])    # bound on f_2(x)
```

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Running the experiments

Run each script from the repository root:

```bash
python vvRKHS_2d_partial.py
```

| Script | Experiment | Saved figures |
|---|---|---|
| `vvRKHS_2d_partial.py` | Observing `f_1`, `f_1 + f_2`, or alternating components of a 2-D output | none (shown on screen) |
| `vvRKHS_2d_partial_nonseparable.py` | Sensitivity of the bound to `gamma` and `delta_ub` with a non-separable kernel | `1_input_2_output_3.pdf` |
| `vvRKHS_nd_partial.py` | Learning 20-step Van der Pol trajectories from a few observed time steps | `2_input_20_output_3.png`, `2_input_20_output_5.pdf` |
| `vvRKHS_regression_gp.py` | Function-valued output learned from point evaluations | `1_input_RKHS_output_1.pdf` |

Additional figures are available as commented-out calls at the bottom of each script.
