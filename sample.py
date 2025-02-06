from similarity_search.loaders.image import DirectoryLoader
from similarity_search.db import SQLLiteDB
from similarity_search.engine import FaissEngine
from similarity_search import Model
from similarity_search.plotting import ImagePlotter

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from pprint import pprint
import cv2

def preprocess_image(arr):
    img = cv2.resize(arr, (299, 299))
    if len(img.shape) == 2:  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return preprocess_input(img)

base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

model = Model(base_model, preprocess=preprocess_image)

db = SQLLiteDB("mydb.db")

loader = DirectoryLoader(base_dir="assets/images", database=db, batch_size=32, excluded_fields=['dir_name', 'image_name'])
# loader.load_to_db(table_name="test", model=model)

engine = FaissEngine(model, db, "test")


query_image_path = "assets/images/baboon/n02486410_620.JPEG"
img = image.load_img(query_image_path, target_size=(299, 299))
img = image.img_to_array(img)

meta = engine.search(img, 6)

pprint(meta)
data = loader.fetch_data(meta)

ImagePlotter.plot_fetched_images(data, show_info=["distance"])

