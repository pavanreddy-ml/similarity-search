from abc import ABC, abstractmethod

from ise.db import Database

class Engine(ABC):
    def __init__(self, 
                 database: Database, 
                 *args, 
                 **kwargs):
        self.database = database

    @abstractmethod
    def initialize(self, 
                   table_name: str, 
                   primary_key: str,
                   embeddings_column: str):
        pass

    @abstractmethod
    def search(self):
        pass