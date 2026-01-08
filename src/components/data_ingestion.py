import io
import sys
import pandas as pd
import numpy as np

import gridfs

from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        try:
            client = MongoClient("mongodb://localhost:27017/")
            self.db = client["clarityAI_database"]
            self.fs = gridfs.GridFS(self.db)
            logging.info("MongoDB GridFS connection established")
        except Exception as e:
            raise CustomException(e, sys)

    def store_file(self, file):
        """
        Stores uploaded file in MongoDB GridFS
        """
        try:
            self.fs.put(
                file.read(),
                filename=file.filename,
                content_type=file.content_type
            )
            logging.info(f"File {file.filename} stored in GridFS")
        except Exception as e:
            raise CustomException(e, sys)
    def get_preview(self):
        """
        Reads latest uploaded file and returns first 5 rows + columns
        """
        try:
            latest_file = self.fs.find().sort("uploadDate", -1).limit(1)

            for file in latest_file:
                file_bytes = file.read()
                df = pd.read_csv(io.BytesIO(file_bytes))
                return df.head().values.tolist(), df.columns.tolist()

            return None, None
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_all_filenames(self):
        return sorted(
            self.db.fs.files.distinct("filename")
        )
