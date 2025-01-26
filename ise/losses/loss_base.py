from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def calculate(self, 
                  x: np.ndarray, 
                  array: np.ndarray, 
                  *args, 
                  **kwargs
                  ) -> np.ndarray:
        pass

    def __call__(self, 
                  x: np.ndarray, 
                  array: np.ndarray, 
                  *args, 
                  **kwargs
                  ) -> np.ndarray:
        return self.calculate(x, array, *args, **kwargs)