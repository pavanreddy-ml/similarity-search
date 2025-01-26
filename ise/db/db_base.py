from abc import ABC, abstractmethod
import uuid
import numpy as np

from ise.constants import DATASAMPLE_KEY, VECTOR_EMBEDDING_KEY

class Database(ABC):
    _column_types  = {}

    @property
    def column_types(self):
        return self._column_types 
    
    def __init__(self,
                 primary_key,
                 vector_embedding_key,
                 *args,
                 **kwargs):
        self.primary_key = primary_key
        self.vector_embedding_key = vector_embedding_key

    @abstractmethod
    def connect(self, *args, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def delete_table(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_table(self, schema, *args, **kwargs):
        pass
    
    @abstractmethod
    def insert(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def batch_insert(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_vectors(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_data_types(self, *args, **kwargs):
        pass

    def infer_schema(self, data, include_datapoint=False):
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TypeError("Data must be a list of dictionaries")
        
        schema = {}
        acceptable_types = list(self.column_types.keys())

        for col in data[0]:
            if type(data[0][col]) in acceptable_types:
                schema[col] = self.column_types[type(data[0][col])]

        if self.primary_key not in schema:
            schema[self.primary_key] = self.column_types[uuid.UUID]

        schema[VECTOR_EMBEDDING_KEY] = self.column_types[np.ndarray]

        if not include_datapoint:
            schema.pop(DATASAMPLE_KEY, None)

        return schema