from ise.loaders.image import DirectoryLoader
from ise.db import SQLLiteDB
from ise.engine import FaissEngine
from ise.constants import Constants

from pprint import pprint

loader = DirectoryLoader(base_dir="assets/images")

samples = loader.get_samples(10)

db = SQLLiteDB("mydb.db")
db.connect()
schema = db.infer_schema(samples)

db.create_table("test", schema, True)

for batch in loader:
    for i in batch:
        i.pop(Constants.DATASAMPLE_KEY, None)
        i[Constants.VECTOR_EMBEDDING_KEY] = [1, 2, 3]
    
    db.batch_insert("test", batch)

engine = FaissEngine(db)
engine.initialize("test", Constants.PRIMARY_KEY, Constants.VECTOR_EMBEDDING_KEY)
pprint(engine.search([1, 2, 3], 3))
