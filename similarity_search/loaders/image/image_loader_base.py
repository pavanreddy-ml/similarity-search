from abc import ABC, abstractmethod

from similarity_search.loaders.loader_base import DataLoader
from similarity_search.db import Database

import numpy as np
from PIL import Image
import concurrent.futures
from copy import deepcopy

from typing import List, Dict, Any

class ImageDataLoader(DataLoader):
    def __init__(self, 
                 database: Database,
                 batch_size: int = 32
                 ) -> None:
        super().__init__(database=database, batch_size=batch_size)

    def _load_batch(self, 
                   metadata_list: List[Dict[str, Any]],
                    *args, 
                    **kwargs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: self._load(p, *args, **kwargs), metadata_list))
        return results

    def _load(self, 
             metadata: Dict[str, Any],
             *args, 
             **kwargs
             ) -> Dict[str, Any]:
        try:
            with Image.open(metadata["absolute_image_path"]) as img:
                copy_of_meta = deepcopy(metadata)
                copy_of_meta[self.database.datasample_key] = np.array(img)
                return copy_of_meta
        except Exception as e:
            print(f"Error loading image {metadata}: {e}")
            return None

    def get(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented as needed.")
