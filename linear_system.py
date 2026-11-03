import numpy as np
from regression.tools import Tools

# ── System dynamics ────────────────────────────────────────────────────────────

def _linear_rotation(x: np.ndarray, r: float = 0.94, theta: float = np.pi / 6) -> np.ndarray:
    A = r * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    return A @ x

def _van_der_pol(x: np.ndarray, mu: float = -0.5, dt: float = 0.4) -> np.ndarray:
    def f(x):
        return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])
    k1 = f(x);              k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2); k4 = f(x +     dt*k3)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def _duffing(x: np.ndarray, alpha: float = -1.0, beta: float = 1.0,
             delta: float = 0.2, dt: float = 0.2) -> np.ndarray:
    """Duffing oscillator: x1' = x2,  x2' = -delta*x2 - alpha*x1 - beta*x1^3."""
    def f(x):
        return np.array([x[1], -delta*x[1] - alpha*x[0] - beta*x[0]**3])
    k1 = f(x);              k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2); k4 = f(x +     dt*k3)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# dt=None means the system is a discrete map — fu is added without scaling
SYSTEMS = {
    "linear_rotation": (_linear_rotation, None),
    "van_der_pol":     (_van_der_pol,     0.05),
    "duffing":         (_duffing,         0.05),
}

C_DEFAULT = np.array([0.5, 0.5])
U_GRID    = np.linspace(0, 1, 10).reshape(-1, 1)
_RNG      = np.random.default_rng(seed=42)
_ALPHA    = _RNG.uniform(-1, 1, (10, 2)) * 2.25         # fixed weights, shape (10, 2)


# ── Simulator ──────────────────────────────────────────────────────────────────

def simulate(
    x0: np.ndarray,
    u: np.ndarray,
    length_scale: float = 1.0,
    system: str = "linear_rotation",
    N: int = 20,
) -> np.ndarray:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system '{system}'. Choose from {list(SYSTEMS)}")

    g, dt = SYSTEMS[system]
    x     = np.asarray(x0, dtype=float).reshape(2)
    u_    = np.atleast_1d(u).reshape(1, -1)

    K_uU = Tools.Kernel.RBF(u_, U_GRID, length_scale=length_scale)  # (1, 10)
    fu   = np.squeeze(K_uU @ _ALPHA)                           # (2,)
    fu_scaled = fu * dt if dt is not None else fu

    X = np.empty((N, 2))
    y = np.empty(N)
    X[0] = x
    y[0] = float(C_DEFAULT @ x)
    for k in range(N - 1):
        x        = g(x) + fu_scaled
        X[k+1]   = x
        y[k+1]   = float(C_DEFAULT @ x)

    return y, X