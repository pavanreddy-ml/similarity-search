import os

from . import ImageDataLoader

from typing import List, Dict, Any

from similarity_search.db import Database


class DirectoryLoader(ImageDataLoader):
    def __init__(self, base_dir: str, database: Database, batch_size=32, excluded_fields=None):
        super().__init__(database=database, batch_size=batch_size)
        self.base_dir = os.path.abspath(base_dir)
        self.excluded_fields = excluded_fields or []

        if "image_path" in self.excluded_fields or "absolute_image_path" in self.excluded_fields:
            raise ValueError('image_path" and "absolute_image_path" cannot be excluded')

    def get_ids(self) -> List[Dict[str, Any]]:
        metadata = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    absolute_path = os.path.abspath(os.path.join(root, file))
                    relative_path = os.path.relpath(absolute_path, self.base_dir)
                    dir_name = os.path.basename(os.path.dirname(absolute_path))

                    metadata.append({
                        "image_name": file,
                        "absolute_image_path": absolute_path,
                        "image_path": relative_path,
                        "dir_name": dir_name
                    })

        for field in self.excluded_fields:
            for item in metadata:
                item.pop(field, None)

        return metadata
    
    def fetch_data(self, metadata, *args, **kwargs):
        data = []
        for item in metadata:
            data_point = self._load(item)
            data_point["image"] = data_point.pop(self.database.datasample_key, None)
            data.append(data_point)

        return data