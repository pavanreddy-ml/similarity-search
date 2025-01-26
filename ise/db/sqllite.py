import json
import sqlite3
import numpy as np
from copy import deepcopy
import uuid

from ise.db import Database


class SQLLiteDB(Database):
    _column_types  = {
        int: "INTEGER",
        float: "REAL",
        str: "TEXT",
        list: "TEXT",
        dict: "TEXT",
        np.ndarray: "TEXT",
        uuid.UUID: "TEXT"
    }

    def __init__(self, 
                 path, 
                 primary_key,
                 vector_embedding_key,
                 *args, 
                 **kwargs):
        super().__init__(primary_key=primary_key, vector_embedding_key=vector_embedding_key *args, **kwargs)
        self.db_path = path
        self.connection = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise Exception(f"Failed to connect to SQLite database: {e}")

    def create_table(self, 
                     table_name, 
                     schema, 
                     force_create=False):
        if not self.connection:
            raise Exception("Database connection is not established")

        if force_create:
            self.delete_table(table_name)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        result = cursor.fetchone()
        if result is not None:
            raise ValueError(f"Table '{table_name}' already exists")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")
        
        if not hasattr(self, "primary_key") or self.primary_key not in schema:
            raise ValueError(f"Primary key '{self.primary_key}' is not defined in the schema")
        
        columns = [
            f"{col} {dtype}{' PRIMARY KEY' if col == self.primary_key else ''}"
            for col, dtype in schema.items()
        ]
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"

        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_query)
            self.connection.commit()
        except sqlite3.Error as e:
            raise Exception(f"Failed to create table: {e}")

    def delete_table(self, 
                     table_name):
        if not self.connection:
            raise Exception("Database connection is not established")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        delete_table_query = f"DROP TABLE IF EXISTS {table_name};"

        try:
            cursor = self.connection.cursor()
            cursor.execute(delete_table_query)
            self.connection.commit()
        except sqlite3.Error as e:
            raise Exception(f"Failed to delete table: {e}")

    def insert(self, 
               table_name, 
               data_point):
        if not self.connection:
            raise Exception("Database connection is not established")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")
        
        if not isinstance(data_point, dict):
            raise TypeError("Data Point must be a dictionarie")
        
        processed_data_point = self.process_data_types([deepcopy(data_point)])[0]

        columns = ', '.join(processed_data_point.keys())
        placeholders = ', '.join(['?' for _ in processed_data_point.values()])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_query, tuple(processed_data_point.values()))
            self.connection.commit()
        except sqlite3.Error as e:
            raise Exception(f"Failed to insert data: {e}")

    def batch_insert(self, 
                     table_name, 
                     data):
        if not self.connection:
            raise Exception("Database connection is not established")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TypeError("Data must be a list of dictionaries")
        
        if not data:
            raise ValueError("Data is empty")
        
        processed_data = self.process_data_types(deepcopy(data))

        columns = ', '.join(processed_data[0].keys())
        placeholders = ', '.join(['?' for _ in processed_data[0].values()])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        values = [tuple(item.values()) for item in processed_data]

        try:
            cursor = self.connection.cursor()
            cursor.executemany(insert_query, values)
            self.connection.commit()
        except sqlite3.Error as e:
            raise Exception(f"Failed to batch insert data: {e}")
        
    def fetch(self, table_name, ids):
        if not self.connection:
            raise Exception("Database connection is not established")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        if not isinstance(ids, list) or not ids:
            raise ValueError("The 'ids' parameter must be a non-empty list")

        placeholders = ', '.join(['?' for _ in ids])
        query = f"SELECT * FROM {table_name} WHERE {self.primary_key} IN ({placeholders})"

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, tuple(ids))
            records = cursor.fetchall()

            # Fetch column names
            column_names = [description[0] for description in cursor.description]

            # Convert rows to a list of dictionaries
            result = [dict(zip(column_names, row)) for row in records]
            return result
        except sqlite3.Error as e:
            raise Exception(f"Failed to fetch records: {e}")
        
    def fetch_vectors(self, table_name):
        if not self.connection:
            raise Exception("Database connection is not established")

        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        query = f"SELECT {self.primary_key}, {self.vector_embedding_key} FROM {table_name}"

        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            records = cursor.fetchall()
            result = {
                record[0]: json.loads(record[1]) if record[1] else []
                for record in records
            }
            return result
        except sqlite3.Error as e:
            raise Exception(f"Failed to fetch vectors: {e}")

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def process_data_types(self, 
                           data):
        for data_point in data:
            if self.primary_key not in data_point:
                data_point[self.primary_key] = str(uuid.uuid4())

        for data_point in data:
            for key, value in data_point.items():
                if isinstance(value, np.ndarray):
                    data_point[key] = json.dumps(value.tolist())
                if isinstance(value, (list, dict)):
                    data_point[key] = json.dumps(value)
                if isinstance(value, uuid.UUID):
                    data_point[key] = str(value)
        return data
