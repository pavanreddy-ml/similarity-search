from ise.losses import Loss

import numpy as np


class CosineSimilarity(Loss):
    def __init__(self):
        pass
    
    def calculate(self, 
                  x: np.ndarray, 
                  array: np.ndarray, 
                  *args, 
                  **kwargs
                  ) -> np.ndarray:
        return np.dot(x, array) / (np.linalg.norm(x) * np.linalg.norm(array))
        