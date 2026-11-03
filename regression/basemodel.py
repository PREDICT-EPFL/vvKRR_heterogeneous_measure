from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

Array2D = NDArray[np.float64]
Array1D = NDArray[np.float64]

class BaseModel(ABC):
    def __init__(self, X: Array2D, Y: Array1D):
        pass

    def fit(self, X, Y) -> None:
        raise NotImplementedError

    def get_mean(self) -> float:
        raise NotImplementedError

    def get_bounds(self) -> Array1D:
        raise NotImplementedError

    def get_next(self) -> tuple[Array1D, float]:
        raise NotImplementedError

    def get_final(self) -> Array1D:
        raise NotImplementedError
    
    def plot(self):
        raise NotImplementedError