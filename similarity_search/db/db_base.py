from abc import ABC, abstractmethod
import uuid
import numpy as np

from similarity_search.constants import Constants

class Database(ABC):
    _column_types  = {}

    @property
    def column_types(self):
        return self._column_types 
    
    def __init__(self,
                 primary_key=None,
                 vector_embedding_key=None,
                 datasample_key=None,
                 *args,
                 **kwargs):
        self.primary_key = Constants.PRIMARY_KEY if primary_key is None else primary_key
        self.vector_embedding_key = Constants.VECTOR_EMBEDDING_KEY if vector_embedding_key is None else vector_embedding_key
        self.datasample_key = Constants.DATASAMPLE_KEY if datasample_key is None else datasample_key
        self.schema = None

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
    def get_columns(self, *args, **kwargs):
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

    def get_schema(self):
        if self.schema is not None:
            return self.schema
        
        self.schema = self.get_columns()
        return self.schema

    def infer_schema_from_data(self, 
                     data, 
                     include_datapoint=False):
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TypeError("Data must be a list of dictionaries")
        
        schema = {}
        acceptable_types = list(self.column_types.keys())

        for col in data[0]:
            if type(data[0][col]) in acceptable_types:
                schema[col] = self.column_types[type(data[0][col])]

        if self.primary_key not in schema:
            schema[self.primary_key] = self.column_types[uuid.UUID]

        schema[self.vector_embedding_key] = self.column_types[np.ndarray]

        if not include_datapoint:
            schema.pop(self.datasample_key, None)

        return schema