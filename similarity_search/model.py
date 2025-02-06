import numpy as np

class Model:
    def __new__(cls, model, preprocess=None, *args, **kwargs):
        instance = super().__new__(cls)

        model_type = type(model).__name__

        if callable(preprocess):
            instance.preprocess = preprocess
        elif preprocess is None:
            instance.preprocess = lambda x: x
        else:
            raise ValueError("Preprocess function must be callable or None")

        if model_type == "Module":
            import torch
            instance.predict = instance.predict_torch
            instance.model = model
            instance.model.eval()
        elif model_type == "Sequential" or hasattr(model, "predict"):
            import tensorflow as tf
            instance.predict = instance.predict_tf
            instance.model = model
        elif callable(model):
            instance.predict = lambda x: model(instance.preprocess(x))
            instance.model = None
        else:
            raise ValueError("Model framework not supported")
        return instance

    def __init__(self, model, preprocess=None, *args, **kwargs):
        self.model = model

    def predict_tf(self, x):
        preprocessed_x = self.handle_batch_preprocess(x)
        return self.model.predict(preprocessed_x)

    def predict_torch(self, x):
        preprocessed_x = self.handle_batch_preprocess(x)
        import torch
        with torch.no_grad():
            return self.model(preprocessed_x)
        
    def predict(self, x):
        pass

    def preprocess(self, arr):
        return arr
    
    def handle_batch_preprocess(self, x):
        if not isinstance(x, list):
            raise ValueError("Input must be a list of arrays")
        
        preprocessed_x = []
        for item in x:
            preprocessed_x.append(self.preprocess(item))
        return np.array(preprocessed_x)

