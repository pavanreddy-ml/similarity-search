

class Model():
    def __new__(cls, model, *args, **kwargs):
        instance = super().__new__(cls)

        if repr(model) == 'torch.nn.modules.module.Module':
            import torch
            instance.predict = instance.predict_torch
        elif repr(model) == 'keras.src.models.model.Model':
            import tensorflow as tf
            instance.predict = instance.predict_tf
        else:
            raise ValueError('Model Framework not supported')
        
        return instance
    
    def __init__(self, model):
        self.model = model

    def predict_tf(self, x):
        return self.model.predict(x)

    def predict_torch(self, x):
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.model(x)