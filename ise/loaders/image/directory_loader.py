import os

from . import ImageDataLoader

from typing import List, Dict, Any


class DirectoryLoader(ImageDataLoader):
    def __init__(self, base_dir: str, batch_size=32):
        super().__init__(batch_size)
        self.base_dir = os.path.abspath(base_dir)

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

        return metadata