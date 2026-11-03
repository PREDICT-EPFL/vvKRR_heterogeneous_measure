from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

Array2D = NDArray[np.float64]
Array1D = NDArray[np.float64]

class Tools:

    class Kernel:
        '''
        Input for kernel is 2D array, where row represents samples, the columns represents problem dimension (features)
        Output for kernel is 2D array, in case of scalar, use np.squeeze()
        '''
        @staticmethod
        def linear(X_1: NDArray, X_2: Optional[NDArray] = None) -> NDArray:
            if X_2 is None:
                X_2 = X_1
            return X_1 @ X_2.T
        
        @staticmethod
        def RBF(X_1: NDArray, X_2: Optional[NDArray] = None, length_scale: Union[float, Array1D] = 1.0) -> NDArray:
            if X_2 is None:
                X_2 = X_1
            if np.array(length_scale).ndim != 0 and np.shape(np.array(length_scale))[0] != np.shape(X_1)[1]:
                raise ValueError('Dimension of length_scale does not match the problem dimension!')
            
            X_1_scaled = X_1/length_scale
            X_2_scaled = X_2/length_scale
            
            sq_dists = np.sum(X_1_scaled**2, axis=1).reshape(-1, 1) + np.sum(X_2_scaled**2, axis=1) - 2 * X_1_scaled @ X_2_scaled.T
            return np.exp(-0.5 * sq_dists)

        @staticmethod
        def polynomial(X_1: NDArray, X_2: Optional[NDArray] = None, degree: int = 2, coef0: float = 1.0) -> NDArray:
            if X_2 is None:
                X_2 = X_1
            return (X_1 @ X_2.T + coef0) ** degree

        @staticmethod
        def matern32(X: NDArray, X_2: Optional[NDArray] = None, length_scale: float = 1.0) -> NDArray:
            pass

    class Kernel4Ca:
        '''
        the one of the input must be ca.MX
        the shape of ca.MX, which represent a sample, must be (1,d), if n samples, ca.MX.shape = (n,d)
        '''
        @staticmethod
        def linear(X_1: Union[ca.MX,NDArray], X_2: Optional[Union[ca.MX,NDArray]] = None) -> ca.MX:
            if X_2 is None:
                X_2 = X_1
            return X_1 @ X_2.T
        
        @staticmethod
        def RBF(X_1: Union[ca.MX,NDArray], X_2: Optional[Union[ca.MX,NDArray]] = None, length_scale: float = 1.0) -> ca.MX:
            # same for NDArray and ca.MX. ca will transfer np.array to ca.DM
            if X_2 is None:
                X_2 = X_1
            if np.array(length_scale).ndim != 0 and np.shape(np.array(length_scale))[0] != np.shape(X_1)[1]:
                raise ValueError('Dimension of length_scale does not match the problem dimension!')
            if np.array(length_scale).ndim == 0:
                X_1_scaled = X_1/length_scale
                X_2_scaled = X_2/length_scale
            else:
                X_1_scaled = X_1.T/length_scale
                X_2_scaled = X_2.T/length_scale

                X_1_scaled = X_1_scaled.T
                X_2_scaled = X_2_scaled.T

            sq_dists = ca.repmat(ca.sum2(X_1_scaled**2),1,X_2_scaled.shape[0]) + ca.repmat(ca.sum2(X_2_scaled**2).T,X_1_scaled.shape[0],1) - 2 * X_1_scaled @ X_2_scaled.T
            return ca.exp(-0.5 * sq_dists)

        @staticmethod
        def polynomial(X_1: Union[ca.MX,NDArray], X_2: Optional[Union[ca.MX,NDArray]] = None, degree: int = 2, coef0: float = 1.0) -> ca.MX:
            pass

        @staticmethod
        def matern32(X_1: Union[ca.MX,NDArray], X_2: Optional[Union[ca.MX,NDArray]] = None, length_scale: float = 1.0) -> ca.MX:
            pass

    class others:
        @staticmethod
        def make_grid(bounds: Array2D, density: Union[int, list]) -> Array2D:
            '''
            Given the bounds, generate a 2D array where each row represents a ND point
            '''
            ndim = bounds.shape[0]
            if isinstance(density, int):
                density = np.full(ndim, density)
            elif ndim == len(density):
                density = np.array(density)
            else:
                raise ValueError('Dimension of density does not match bound.')

            axes = []
            for (lo, hi), d in zip(bounds, density):
                 n = d + 1
                 axes.append(np.linspace(lo, hi, n))

            mesh = np.meshgrid(*axes,indexing = 'ij')
            grid = np.stack([m.ravel() for m in mesh], axis=-1)

            return grid
        
        @staticmethod
        def plot_contour(grid: Array2D, value: Array1D) -> None:
            plt.figure(figsize=(6,6))
            x = grid[:, 0]
            y = grid[:, 1]
            z = value
            
            # Create a grid and interpolate scattered data if needed
            xi = np.linspace(min(x), max(x), 100)
            yi = np.linspace(min(y), max(y), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata(grid, z, (Xi, Yi), method='linear')

            cs = plt.contourf(Xi, Yi, Zi, cmap='viridis', levels=40,zorder=0)
            # plt.xlabel('x_1')
            # plt.ylabel('x_2')
            # plt.title('Contour plot')
            # plt.colorbar(cs)

            idx_max = np.argmax(z)
            x_max, y_max = x[idx_max], y[idx_max]

