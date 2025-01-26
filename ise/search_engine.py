from ise import Model
from typing import Union
from ise.losses import Loss, LossRegistry
from ise.loaders import DataLoader
from ise.engine import Engine

class SimilaritySearchEngine:
    def __init__(self,
                 model,
                 data_loader: DataLoader,
                 engine: str | Engine,
                 loss: Union[str, Loss]
                 ):
        self.model = Model(model=model)
        self.loss = LossRegistry(loss)

        if not isinstance(data_loader, DataLoader):
            raise ValueError("data_loader must be an instance of DataLoader")
        self.data_loader = data_loader



    def create_index(self, data):
        pass

    def search(self, 
               sample, 
               top_k=1, 
               post_search_filter=None):
        pass

