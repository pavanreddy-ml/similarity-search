from similarity_search.engine import Engine
from similarity_search.db import Database

import faiss
import numpy as np

class FaissEngine(Engine):
    def __init__(self, 
                 model,
                 database: Database, 
                 table_name: str, 
                 *args, 
                 **kwargs):
        super().__init__(model, database, *args, **kwargs)
        self.table_name = table_name
        self.initialize()

    def initialize(self) -> None:
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

        query_vector = [np.array(query_vector, dtype=np.float32)]
        embedding = self.model.predict(query_vector).reshape(1, -1)
        faiss.normalize_L2(embedding)

        distances, indices = self.index.search(embedding, k)
        distances = distances[0]
        indices = indices[0]

        ids = [self.id_map[idx] for idx in indices if idx != -1]

        records = self.database.fetch(self.table_name, ids)

        results = [{
            "distance": distances[i],
            **records[i]
        } for i in range(len(distances))]

        for result in results:
            result.pop(self.database.vector_embedding_key, None)

        return results
    