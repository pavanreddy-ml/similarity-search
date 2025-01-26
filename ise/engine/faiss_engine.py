from ise.engine import Engine

import faiss
import numpy as np

class FaissEngine(Engine):
    def __init__(self, database, table_name, *args, **kwargs):
        super().__init__(database, *args, **kwargs)
        self.table_name = table_name

    def initialize(self):
        vectors = self.database.fetch_vectors(self.table_name)

        if not vectors:
            raise ValueError("No vectors found in the database to initialize the FAISS index.")

        ids = list(vectors.keys())
        embeddings = np.array(list(vectors.values()), dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        dimension = embeddings.shape[1]

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.id_map = {i: ids[i] for i in range(len(ids))}

    def search(self, query_vector, k=10):
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Call `initialize` first.")

        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)
        distances = distances[0]
        indices = indices[0]

        ids = [self.id_map[idx] for idx in indices if idx != -1]

        records = self.database.fetch(self.table_name, ids)

        results = [{
            "distances": distances[i],
            **records[i]
        } for i in range(len(distances))]


        return results