"""Kernel regression models used by the vvRKHS experiments.

Trimmed copy of the model classes from https://github.com/Wenbin24/regression.
Only the classes needed by the experiment scripts are kept; their code is unchanged.
"""
from .basemodel import BaseModel
from numpy.typing import NDArray
from typing import Callable, Tuple, Union
from .tools import Tools
import casadi
import numpy as np

# np.random.seed(1)

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Array3D = NDArray[np.float64]
'''
General data structure: X: Array2D, (number of sapmles, problem dimension), Y: Array1D, (number of samples)
The new sample is always append from the end
The input and output for kernel is always 2D array
The output must be squeezed if a scalar is wanted, since np.array([[1,1,1]])@np.array([[1],[1],[1]]) = [[3]]
'''
class KRR(BaseModel):
    def __init__(self, 
                 X: Array2D, 
                 Y: Array1D, 
                 kernel: Callable, 
                 length_scale: float, 
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None, 
                 lambda_: float = 1):
        '''
        bound: [[lb_1,ub_1],[lb_2,ub_2]...], each row represent a dimension
        '''
        super().__init__(X, Y)
        self.alpha: Array1D
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.K = kernel(X, length_scale = length_scale)
        self.K_inv = np.linalg.inv(self.K + 0.0001*np.eye(self.n))
        self.bound = bound
        self.kernel = kernel
        self.length_scale = length_scale
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_
        self.fit(X,Y)
        self._get_Delta()
        self._get_norm()

    def fit(self, X: Array2D, Y: Array1D):
        self.alpha = np.linalg.inv(self.K + self.lambda_ * np.eye(self.n)) @ Y
        pass

    def get_mean(self, x: Array2D):
        return np.squeeze(self.kernel(x, self.X, length_scale = self.length_scale) @ self.alpha)

    def get_bound(self, x: Array2D):
        K_xX = self.kernel(x, self.X, length_scale = self.length_scale)
        K_Xx = self.kernel(self.X, x, length_scale = self.length_scale)
        p = np.squeeze(self.kernel(x, x, length_scale = self.length_scale) -  K_xX@ self.K_inv @ K_Xx)
        first_term = np.sqrt(max(p*(self.gamma**2 + self.Delta - self.Y@self.K_inv@self.Y),0))
        second_term = self.delta_ub*np.sum(np.abs(self.K_inv @ K_Xx))
        third_term = np.squeeze(np.abs(K_xX@(np.linalg.inv(self.K + self.lambda_ * np.eye(self.n))-self.K_inv)@self.Y))
        return first_term + second_term + third_term
    
    def get_next(self):
        pass

    def get_final(self):
        pass

    def _get_Delta(self):
        m = casadi.Opti()

        K_inv = casadi.DM(self.K_inv)
        buf = casadi.DM(self.Y@self.K_inv)
        var_d = m.variable(self.n)
        Delta = -var_d.T@K_inv@var_d + 2*buf.T@var_d

        m.subject_to(-var_d <= self.delta_ub)
        m.subject_to(var_d <= self.delta_ub)
        
        m.minimize(-Delta)
        m.solver('ipopt', dict(print_time=False), dict(print_level=False, max_iter=1000))

        sol = m.solve()
        self.Delta = sol.value(Delta)
        pass

    def _get_norm(self):
        self.norm = np.sqrt(self.alpha@self.K@self.alpha)

class KRRNonseparableKernel(BaseModel):
    def __init__(self, 
                 X: Array2D, 
                 Y: Array1D, 
                 length_scale_coeff,
                 length_scale_list,
                 kernel: Callable,
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None, 
                 lambda_: float = 1):
        '''
        Scalar-valued KRR with sum of kernels
        '''
        super().__init__(X, Y)
        self.alpha: Array1D
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.length_scale_coeff = length_scale_coeff
        self.length_scale_list = length_scale_list
        self.kernel = kernel

        self.K = self._kernel_sum(X,X)
        self.K_inv = np.linalg.inv(self.K + 0.0001*np.eye(self.n))

        self.bound = bound
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_
        self.fit(X,Y)
        self._get_Delta()
        self._get_norm()

    def _kernel_sum(self, X1, X2):
        return self.length_scale_coeff[0]*self.kernel(X1,X2, length_scale = self.length_scale_list[0]) + self.length_scale_coeff[1]*self.kernel(X1,X2, length_scale = self.length_scale_list[1])
    
    def fit(self, X: Array2D, Y: Array1D):
        self.alpha = np.linalg.inv(self.K + self.lambda_ * np.eye(self.n)) @ Y
        pass

    def get_mean(self, x: Array2D):
        return np.squeeze(self._kernel_sum(x, self.X) @ self.alpha)

    def get_bound(self, x: Array2D):
        K_xX = self._kernel_sum(x, self.X)
        K_Xx = self._kernel_sum(self.X, x)
        p = np.squeeze(self._kernel_sum(x, x) -  K_xX@ self.K_inv @ K_Xx)
        first_term = np.sqrt(max(p*(self.gamma**2 + self.Delta - self.Y@self.K_inv@self.Y),0))
        second_term = self.delta_ub*np.sum(np.abs(self.K_inv @ K_Xx))
        third_term = np.squeeze(np.abs(K_xX@(np.linalg.inv(self.K + self.lambda_ * np.eye(self.n))-self.K_inv)@self.Y))
        return first_term + second_term + third_term
    
    def get_next(self):
        pass

    def get_final(self):
        pass

    def _get_Delta(self):
        m = casadi.Opti()

        K_inv = casadi.DM(self.K_inv)
        buf = casadi.DM(self.Y@self.K_inv)
        var_d = m.variable(self.n)
        Delta = -var_d.T@K_inv@var_d + 2*buf.T@var_d

        m.subject_to(-var_d <= self.delta_ub)
        m.subject_to(var_d <= self.delta_ub)
        
        m.minimize(-Delta)
        m.solver('ipopt', dict(print_time=False), dict(print_level=False, max_iter=1000))

        sol = m.solve()
        self.Delta = sol.value(Delta)
        pass

    def _get_norm(self):
        self.norm = np.sqrt(self.alpha@self.K@self.alpha)
        
class vvKRR(BaseModel):
    def __init__(self, 
                 X: Array2D,
                 Y: Array2D,
                 kernel: Tools.Kernel,
                 T: Array2D,
                 length_scale: float,
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None,
                 lambda_ = 1):
        '''each row of X represents a data point, each row of Y represent a measurement，
        assume that the kernel is separable'''
        super().__init__(X, Y)
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.xd = X.shape[1]
        self.yd = Y.shape[1]

        self.G = kernel(X, length_scale = length_scale) + 0.0001*np.eye(self.n)
        self.G_eigval, self.G_eigvect = np.linalg.eig(self.G)
        self.G_inv = np.linalg.inv(self.G)

        self.T = T + 0.0001*np.eye(self.yd)
        self.T_eigval, self.T_eigvect = np.linalg.eig(self.T)
        self.T_eigval = self.T_eigval
        self.Y_coordinate_under_T = self.Y@self.T_eigvect

        self.bound = bound
        self.kernel = kernel
        self.length_scale = length_scale
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_
        self.fit(X,Y)
        self._get_Delta()
        self._get_norm()
    
    def fit(self, X: Array2D, Y: Array2D)-> Array2D:
        '''each row of alpha represent the coef. of the corrosponding point'''
        alpha = np.zeros((self.n,self.yd))

        for i in range(self.n):
            # loop for calculating the alpha for each point
            a = np.zeros(self.yd)
            for j in range(self.yd):
                # loop for calculating the coordinate for each alpha
                for k in range(self.n):
                    a[j] += pow(self.G_eigval[k]*self.T_eigval[j] + self.lambda_,-1)*self.G_eigvect[i,k]*self.G_eigvect[:,k].T@self.Y_coordinate_under_T[:,j]
            alpha[i,:] = self.T_eigvect @ a

        self.alpha = alpha
        pass

    def get_mean(self, x: Array2D):
        '''evaluate the mean at give point, the output is a 1D array'''
        k = self.kernel(x, self.X, length_scale = self.length_scale)
        return np.squeeze(self.T@self.alpha.T@k.T)
    
    def get_bound(self, x: Array2D):
        G_xX = self.kernel(x, self.X, length_scale = self.length_scale)
        G_Xx = self.kernel(self.X, x, length_scale = self.length_scale)
        p = np.squeeze(self.kernel(x, x, length_scale = self.length_scale) -  G_xX@ self.G_inv @ G_Xx)
        first_term = np.sqrt(p*np.max(self.T_eigval)*(self.gamma**2+self.Delta - np.trace(self.G_inv@self.Y_coordinate_under_T@np.diag(1/self.T_eigval)@self.Y_coordinate_under_T.T)))
        second_term = np.sum(np.abs(G_xX@self.G_inv))*self.delta_ub
        epsilon = np.zeros((self.n,self.yd))
        for i in range(self.n):
            for j in range(self.yd):
                epsilon[i,j] = (pow(self.G_eigval[i]*self.T_eigval[j] + self.lambda_,-1)-pow(self.G_eigval[i]*self.T_eigval[j],-1))*self.G_eigvect[:,i].T@self.Y_coordinate_under_T[:,j]
        third_term = np.linalg.norm(G_xX@self.G_eigvect@epsilon@np.diag(self.T_eigval))

        # # another way for calculation
        # buf = 0
        # for i in range(self.yd):
        #     buf += np.squeeze(self.T_eigval[i]*G_xX@(np.linalg.inv(self.T_eigval[i]*self.G + self.lambda_*np.eye(self.n)) - np.linalg.inv(self.T_eigval[i]*self.G))@self.Y_coordinate_under_T[:,i])**2

        # print(np.linalg.norm(np.kron(G_xX,self.T)@(np.linalg.inv(np.kron(self.G,self.T) + self.lambda_*np.eye(self.n*self.yd))-np.linalg.inv(np.kron(self.G,self.T)))@self.Y.reshape(-1,1)))
        # print(third_term)
        # print(np.sqrt(buf))
        # print('\n')
        
        return first_term + second_term + third_term
    
    def _get_Delta(self):
        m = casadi.Opti()

        G_inv = casadi.DM(self.G_inv)
        Y_coordinate_under_T = casadi.DM(self.Y_coordinate_under_T)
        omega = m.variable(self.n, self.yd)
        Delta = -casadi.trace(G_inv@omega@casadi.diag(1/self.T_eigval)@omega.T) + 2*casadi.trace(G_inv@Y_coordinate_under_T@casadi.diag(1/self.T_eigval)@omega.T)

        m.subject_to(casadi.sum2(omega**2) <= self.delta_ub**2)
        
        m.minimize(-Delta)
        m.solver('ipopt', dict(print_time=False), dict(print_level=False, acceptable_tol =1e-10, max_iter=2000))

        sol = m.solve()
        self.Delta = sol.value(Delta)
        pass

    def _get_norm(self):
        alpha_coordinate_under_T = self.alpha@self.T_eigvect
        self.norm = np.sqrt(np.trace(self.G@alpha_coordinate_under_T@np.diag(self.T_eigval)@alpha_coordinate_under_T.T))

class vvKRRDirect(BaseModel):
    def __init__(self, 
                 X: Array2D,
                 Y: Array2D,
                 kernel: Tools.Kernel,
                 T: Array2D,
                 length_scale: float,
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None,
                 lambda_ = 1):
        '''each row of X represents a data point, each row of Y represent a measurement,
        assume that the kernel is separable'''
        super().__init__(X, Y)
        self.X = X
        self.Y = Y
        self.Y_concatenate = Y.ravel()
        self.n = X.shape[0]
        self.xd = X.shape[1]
        self.yd = Y.shape[1]

        self.G = kernel(X, length_scale = length_scale) + 0.0001*np.eye(self.n)
        self.G_inv = np.linalg.inv(self.G)
        self.T = T

        self.bound = bound
        self.kernel = kernel
        self.length_scale = length_scale
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_

        self.Gram = np.kron(self.G,self.T)
        self.Gram_inv = np.linalg.inv(self.Gram)
        self.Gram_reg_inv = np.linalg.inv(self.Gram + self.lambda_*np.eye(self.n*self.yd))

        self.fit(X,Y)
        self._get_Delta()
        self._get_norm()
    
    def fit(self, X: Array2D, Y: Array2D)-> Array2D:
        '''each row of alpha represent the coef. of the corrosponding point'''
        self.alpha = self.Gram_reg_inv@self.Y_concatenate
        pass

    def get_mean(self, x: Array2D):
        '''evaluate the mean at give point, the output is a 1D array'''
        k = np.kron(self.kernel(x, self.X, length_scale = self.length_scale),self.T)
        return np.squeeze(k@self.alpha)
    
    def get_bound(self, x: Array2D):
        G_xX = self.kernel(x, self.X, length_scale = self.length_scale)
        G_Xx = self.kernel(self.X, x, length_scale = self.length_scale)
        p = np.squeeze(self.kernel(x, x, length_scale = self.length_scale) -  G_xX@ self.G_inv @ G_Xx)
        first_term = np.sqrt(p*np.linalg.norm(self.T,2)*(self.gamma**2 + self.Delta - self.Y_concatenate@self.Gram_inv@self.Y_concatenate))
        second_term = np.sum(np.abs(G_xX@self.G_inv))*self.delta_ub
        third_term = np.linalg.norm(self.get_mean(x)-np.kron(G_xX,self.T)@self.Gram_inv@self.Y_concatenate)
        return first_term + second_term + third_term
    
    def _get_Delta(self):
        opti = casadi.Opti()

        Gram_inv_dm = casadi.DM(self.Gram_inv)
        y_dm = casadi.DM(self.Y_concatenate.reshape(-1, 1))

        delta = opti.variable(self.n*self.yd, 1)

        Delta = (2 * (delta.T @ Gram_inv_dm @ y_dm)
                - (delta.T @ Gram_inv_dm @ delta))

        # Per-measurement block constraints: ||delta_i||_2 <= delta_ub
        idx = 0
        for i in range(self.n):
            delta_i = delta[idx:idx + self.yd]
            opti.subject_to(casadi.sumsqr(delta_i) <= self.delta_ub**2)
            idx += self.yd

        opti.minimize(-Delta)
        opti.solver('ipopt',
                    dict(print_time=False),
                    dict(print_level=0, acceptable_tol=1e-10, max_iter=2000))

        sol = opti.solve()
        self.Delta = float(sol.value(Delta))
        pass

    def _get_norm(self):
        self.norm = np.sqrt(self.Y_concatenate@self.Gram_inv@self.Y_concatenate)
        
class vvKRRFunctionOutput(BaseModel):
    def __init__(self, 
                 X: Array2D,
                 Y_alpha: Array2D,
                 T_matrix: Array2D,
                 kernel: Tools.Kernel,
                 length_scale: float,
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None,
                 lambda_ = 1):
        super().__init__(X, Y_alpha)
        self.X = X
        self.Y_alpha = Y_alpha
        self.T_matrix = T_matrix
        self.n = X.shape[0]
        self.xd = X.shape[1]

        self.G = kernel(X, length_scale = length_scale) + 0.0001*np.eye(self.n)
        self.G_inv = np.linalg.inv(self.G)

        self.bound = bound
        self.kernel = kernel
        self.length_scale = length_scale
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_
        self._get_Delta()

    def get_mean(self, x: Array2D):
        '''evaluate the mean at given x, the output is the vector of alpha'''
        G_xX = self.kernel(x, self.X, length_scale = self.length_scale)
        return G_xX@np.linalg.inv(self.G+self.lambda_*np.eye(self.n))@self.Y_alpha

    def get_bound(self,  x: Array2D):
        G_xX = self.kernel(x, self.X, length_scale = self.length_scale)
        G_Xx = self.kernel(self.X, x, length_scale = self.length_scale)
        p = np.squeeze(self.kernel(x, x, length_scale = self.length_scale) -  G_xX@ self.G_inv @ G_Xx)
        first_term = np.sqrt(p*(self.gamma**2 + self.Delta - np.trace(self.G_inv@self.Y_alpha@self.T_matrix@self.Y_alpha.T)))
        second_term = np.sum(np.abs(G_xX@self.G_inv))*self.delta_ub
        alpha_buf = G_xX@(np.linalg.inv(self.G + self.lambda_ * np.eye(self.n))-self.G_inv)@self.Y_alpha
        third_term = np.sqrt(np.squeeze(alpha_buf@self.T_matrix@alpha_buf.T))
        return first_term + second_term + third_term
    
    def _get_Delta(self):
        m = casadi.Opti()

        G_inv = casadi.DM(self.G_inv)
        T_matrix = casadi.DM(self.T_matrix)
        Y_alpha = casadi.DM(self.Y_alpha)
        delta_alpha = m.variable(self.n, np.shape(self.T_matrix)[0])
        Delta = -casadi.trace(G_inv@delta_alpha@T_matrix@delta_alpha.T) + 2*casadi.trace(G_inv@Y_alpha@T_matrix@delta_alpha.T)
        
        for i in range(self.n):
            m.subject_to(delta_alpha[i,:]@T_matrix@delta_alpha[i,:].T <= self.delta_ub**2)
        
        m.minimize(-Delta)
        m.solver('ipopt', dict(print_time=False), dict(print_level=False, acceptable_tol =1e-10, max_iter=2000))

        sol = m.solve()
        self.Delta = sol.value(Delta)
        pass

class vvKRRPartial(BaseModel):
    def __init__(self, 
                 X: Array2D,
                 M_list: list,
                 Y: list,
                 kernel: Tools.Kernel,
                 T: Array2D,
                 length_scale: float,
                 gamma: float = 0,
                 delta_ub: float = 0,
                 bound: Array2D = None,
                 lambda_ = 1,
                 if_delta: bool = True,
                 Delta: float = 0,
                 numerical_lambda: int = 4):
        '''each row of X represents a data point, each element in M represent a measurement operator, and each element in Y represent a measurement
        assume that the kernel is separable
        calculate delta can be very slow for high dimensional input (many samples)
        and high dimensional outputs (many measurements for each sample, and cannot explore kron structure)'''
        super().__init__(X, Y)
        self.X = X
        self.M_list = M_list
        self.Y = Y
        self.Y_concatenate = np.concatenate(Y)

        self.n = X.shape[0]
        self.xd = X.shape[1]
        self.M_size_list = [Mi.shape[0] for Mi in self.M_list]
        self.md = sum(self.M_size_list)

        self.G = kernel(X, length_scale = length_scale) + 0.0001*np.eye(self.n)
        self.T = T
        
        self.bound = bound
        self.kernel = kernel
        self.length_scale = length_scale
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_

        self.Gram = self._kernel_with_M(self.X,self.M_list,self.X,self.M_list) + (1/pow(10,numerical_lambda))*np.eye(self.md)
        self.Gram_inv = np.linalg.inv(self.Gram)
        self.Gram_reg = self.Gram + self.lambda_*np.eye(np.shape(self.Gram)[0])
        self.Gram_reg_inv = np.linalg.inv(self.Gram_reg)

        self.fit(X,Y)
        if if_delta:
            self._get_Delta()
        else:
            self.Delta = Delta
        self._get_norm()
    
    def _get_Delta(self):
        opti = casadi.Opti()

        Gram_inv_dm = casadi.DM(self.Gram_inv)
        y_dm = casadi.DM(self.Y_concatenate.reshape(-1, 1))

        delta = opti.variable(self.md, 1)

        Delta = (2 * (delta.T @ Gram_inv_dm @ y_dm)
                - (delta.T @ Gram_inv_dm @ delta))

        # Per-measurement block constraints: ||delta_i||_2 <= delta_ub
        idx = 0
        for mi in self.M_size_list:
            delta_i = delta[idx:idx + mi]
            opti.subject_to(casadi.sumsqr(delta_i) <= self.delta_ub**2)
            idx += mi

        opti.minimize(-Delta)
        opti.solver('ipopt',
                    dict(print_time=False),
                    dict(print_level=0, acceptable_tol=1e-10, max_iter=2000))

        sol = opti.solve()
        self.Delta = float(sol.value(Delta))
        pass

    def _get_nabla_optimal(self, x: Array2D, M_query:list):
        MK_xXM_list = self._kernel_with_M(x,M_query,self.X,self.M_list)
        A = MK_xXM_list @ self.Gram_inv

        A_dm = casadi.DM(A)

        best_nabla = 0.0
        rng = np.random.default_rng(0)

        for _ in range(10):  # multiple restarts for global max
            opti = casadi.Opti()
            delta = opti.variable(self.md, 1)

            nabla_sq = casadi.sumsqr(A_dm @ delta)  # ||A delta||^2

            # Per-block constraints
            idx = 0
            for mi in self.M_size_list:
                opti.subject_to(casadi.sumsqr(delta[idx:idx + mi]) <= self.delta_ub**2)
                idx += mi

            opti.minimize(-nabla_sq)
            opti.solver('ipopt',
                        dict(print_time=False),
                        dict(print_level=0, acceptable_tol=1e-10, max_iter=2000))

            # Random initialization on the boundary of each block
            delta0 = np.zeros(self.md)
            idx = 0
            for mi in self.M_size_list:
                v = rng.standard_normal(mi)
                delta0[idx:idx + mi] = self.delta_ub * v / np.linalg.norm(v)
                idx += mi
            opti.set_initial(delta, delta0)

            try:
                sol = opti.solve()
                val = float(np.sqrt(sol.value(nabla_sq)))
                if val > best_nabla:
                    best_nabla = val
            except Exception:
                pass

        return best_nabla
    
    def _get_nabla_approx(self, x: Array2D, M_query: list):
        MK_xXM_list = self._kernel_with_M(x, M_query, self.X, self.M_list)
        A = MK_xXM_list @ self.Gram_inv  # (m_query, md)

        nabla = 0.0
        idx = 0
        for mi in self.M_size_list:
            A_i = A[:, idx:idx + mi]                  # column block for sample i
            nabla += np.linalg.norm(A_i, ord=2)       # spectral norm ||A_i||_op
            idx += mi

        return nabla * self.delta_ub

    def _kernel_with_M(
            self,
            X_1: Array2D,
            M_1: list,
            X_2: Array2D,
            M_2: list,
        ):
            G = self.kernel(X_1, X_2, length_scale = self.length_scale)

            n1 = len(M_1)
            n2 = len(M_2)

            blocks = [
                [
                    G[i, j] * (M_1[i] @ self.T @ M_2[j].T)
                    for j in range(n2)
                ]
                for i in range(n1)
            ]

            return np.block(blocks)
    
    def fit(self, X, Y):
        self.alpha = self.Gram_reg_inv@self.Y_concatenate

    def get_mean(self, x:Array2D, M_query:list):
        MK_xXM_list = self._kernel_with_M(x,M_query,self.X,self.M_list)
        return MK_xXM_list@self.alpha
    
    def get_bound(self,x:Array2D,M_query:list):
        MK_xxM = self._kernel_with_M(x,M_query,x,M_query)
        MK_xXM_list = self._kernel_with_M(x,M_query,self.X,self.M_list)
        M_listK_XxM = self._kernel_with_M(self.X,self.M_list,x,M_query)
        P_M = MK_xxM - MK_xXM_list@self.Gram_inv@M_listK_XxM
        eig_P_M,_ = np.linalg.eigh(P_M)
        p = np.sqrt(max(eig_P_M))
        first_term = p*np.sqrt(self.gamma**2 + self.Delta - self.Y_concatenate.T@self.Gram_inv@self.Y_concatenate)
        second_term = self._get_nabla_approx(x,M_query)
        third_term = np.linalg.norm(MK_xXM_list@self.Gram_inv@self.Y_concatenate - self.get_mean(x,M_query))
        
        return first_term + second_term + third_term
        # return p

    def _get_norm(self):
        self.norm = np.sqrt(self.Y_concatenate.T@self.Gram_inv@self.Y_concatenate)
        
class vvKRRPartialNonseparableKernel(BaseModel):
    def __init__(self, 
                X: Array2D,
                M_list: list,
                Y: list,
                theta:float,
                kernel: Tools.Kernel,
                gamma: float = 0,
                delta_ub: float = 0,
                bound: Array2D = None,
                lambda_ = 1,
                if_delta: bool = True,
                Delta: float = 0,
                numerical_lambda: int = 4):
        super().__init__(X, Y)
        self.X = X
        self.M_list = M_list
        self.Y = Y
        self.Y_concatenate = np.concatenate(Y)
        self.theta = theta

        self.n = X.shape[0]
        self.xd = X.shape[1]
        self.M_size_list = [Mi.shape[0] for Mi in self.M_list]
        self.md = sum(self.M_size_list)
        
        self.bound = bound
        self.kernel = kernel
        self.gamma = gamma
        self.delta_ub = delta_ub
        self.lambda_ = lambda_

        self.Gram = self._kernel_with_M(self.X,self.M_list,self.X,self.M_list) + (1/pow(10,numerical_lambda))*np.eye(self.md)
        self.Gram_inv = np.linalg.inv(self.Gram)
        self.Gram_reg = self.Gram + self.lambda_*np.eye(np.shape(self.Gram)[0])
        self.Gram_reg_inv = np.linalg.inv(self.Gram_reg)

        self.fit(X,Y)
        if if_delta:
            self._get_Delta()
        else:
            self.Delta = Delta
        self._get_norm()
    
    def _get_Delta(self):
        opti = casadi.Opti()

        Gram_inv_dm = casadi.DM(self.Gram_inv)
        y_dm = casadi.DM(self.Y_concatenate.reshape(-1, 1))

        delta = opti.variable(self.md, 1)

        Delta = (2 * (delta.T @ Gram_inv_dm @ y_dm)
                - (delta.T @ Gram_inv_dm @ delta))

        # Per-measurement block constraints: ||delta_i||_2 <= delta_ub
        idx = 0
        for mi in self.M_size_list:
            delta_i = delta[idx:idx + mi]
            opti.subject_to(casadi.sumsqr(delta_i) <= self.delta_ub**2)
            idx += mi

        opti.minimize(-Delta)
        opti.solver('ipopt',
                    dict(print_time=False),
                    dict(print_level=0, acceptable_tol=1e-10, max_iter=2000))

        sol = opti.solve()
        self.Delta = float(sol.value(Delta))
        pass
    
    def _kernel_with_M(
            self,
            X_1: Array2D,
            M_1: list,
            X_2: Array2D,
            M_2: list,
        ):

            n1 = len(M_1)
            n2 = len(M_2)
            blocks = []

            for i in range(n1):
                blocks.append([])
                for j in range(n2):
                    blocks[i].append((M_1[i] @ self._T(X_1[i].reshape(1,-1),X_2[j].reshape(1,-1)) @ M_2[j].T))
                    
            return np.block(blocks)

    def _T(self,x_1,x_2):
        A = np.array([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        D = np.array([[self.kernel(x_1,x_2,length_scale = 0.05).squeeze(), 0],[0,self.kernel(x_1,x_2,length_scale = 0.5).squeeze()]])
        return A@D@A.T

    def _get_nabla_approx(self, x: Array2D, M_query: list):
        MK_xXM_list = self._kernel_with_M(x, M_query, self.X, self.M_list)
        A = MK_xXM_list @ self.Gram_inv  # (m_query, md)

        nabla = 0.0
        idx = 0
        for mi in self.M_size_list:
            A_i = A[:, idx:idx + mi]                  # column block for sample i
            nabla += np.linalg.norm(A_i, ord=2)       # spectral norm ||A_i||_op
            idx += mi

        return nabla * self.delta_ub
    
    def fit(self, X, Y):
        self.alpha = self.Gram_reg_inv@self.Y_concatenate

    def get_mean(self, x:Array2D, M_query:list):
        MK_xXM_list = self._kernel_with_M(x,M_query,self.X,self.M_list)
        return MK_xXM_list@self.alpha
    
    def get_bound(self,x:Array2D,M_query:list):
        MK_xxM = self._kernel_with_M(x,M_query,x,M_query)
        MK_xXM_list = self._kernel_with_M(x,M_query,self.X,self.M_list)
        M_listK_XxM = self._kernel_with_M(self.X,self.M_list,x,M_query)
        P_M = MK_xxM - MK_xXM_list@self.Gram_inv@M_listK_XxM
        eig_P_M,_ = np.linalg.eigh(P_M)
        p = np.sqrt(max(eig_P_M))
        first_term = p*np.sqrt(self.gamma**2 + self.Delta - self.Y_concatenate.T@self.Gram_inv@self.Y_concatenate)
        second_term = self._get_nabla_approx(x,M_query)
        third_term = np.linalg.norm(MK_xXM_list@self.Gram_inv@self.Y_concatenate - self.get_mean(x,M_query))
        
        return first_term + second_term + third_term
        # return p

    def _get_norm(self):
        self.norm = np.sqrt(self.Y_concatenate.T@self.Gram_inv@self.Y_concatenate)
