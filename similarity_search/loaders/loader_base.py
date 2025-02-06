from abc import ABC, abstractmethod

from similarity_search.db import Database
from similarity_search.model import Model

import numpy as np

class DataLoader(ABC):
    def __init__(self, database: Database, batch_size=32):
        self.batch_counter = 0
        self.database = database
        self.batch_size = batch_size
        self.ids = []

    @abstractmethod
    def get_ids(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def _load_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def _load(self, *args, **kwargs):
        pass
        
    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_data(metadata, *args, **kwargs):
        pass

    def get_samples(self, n=10):
        samples = []
        for batch in self:
            samples.extend(batch)
            if len(samples) >= n:
                return samples[:n]
        return samples
    
    def load_to_db(self, 
                    table_name: str,
                    model: Model, 
                    *args, 
                    **kwargs):
        samples = self.get_samples(5)
        self.schema = self.database.infer_schema_from_data(samples)
        self.database.create_table(table_name, self.schema, True)
        for batch in self:
            data_samples = [i.pop(self.database.datasample_key, None) for i in batch]
            preds = model.predict(data_samples)
            for i, pred in enumerate(preds):
                batch[i][self.database.vector_embedding_key] = pred
            self.database.batch_insert(table_name, batch)

    def __iter__(self):
        self.ids = self.get_ids()
        self.batch_counter = 0
        return self

    def __next__(self):
        start_index = self.batch_size * self.batch_counter
        end_index = min(start_index + self.batch_size, len(self.ids))

        self.batch_counter += 1

        if start_index >= len(self.ids):
            raise StopIteration
        
        return self._load_batch(self.ids[start_index:end_index])