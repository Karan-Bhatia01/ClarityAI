import sys
import io
import pandas as pd

import gridfs
from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException


class DataInfo:
    def __init__(self, filename: str):
        try:
            self.filename = filename

            client = MongoClient("mongodb://localhost:27017/")
            self.db = client["clarityAI_database"]
            self.fs = gridfs.GridFS(self.db)

            logging.info("MongoDB connection established in DataPreprocessing.")

        except Exception as e:
            raise CustomException(e, sys)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Fetch file from Database using filename and return as pandas DataFrame
        """
        try:
            # Get latest file with given filename
            grid_out = self.fs.find_one(
                {"filename": self.filename},
                sort=[("uploadDate", -1)]
            )

            if grid_out is None:
                raise Exception("File not found in MongoDB GridFS")

            file_bytes = grid_out.read()
            df = pd.read_csv(io.BytesIO(file_bytes))

            logging.info("Dataset successfully loaded into DataFrame.")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def dataset_analysis(self) -> dict:
        """
        Perform basic dataset analysis and return results
        """
        try:
            df = self.get_dataframe()

            analysis = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "null_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
                "describe": df.describe().to_dict(),
            }

            logging.info("Dataset analysis completed.")
            return analysis

        except Exception as e:
            raise CustomException(e, sys)

    def get_unique_column_values(self) -> dict:
        """
        Returns a dictionary with column names and its unique values.
        """
        df = self.get_dataframe()
        unique_val = {}
        for col in df.columns:
            unique_val[col]=df[col].unique().tolist()

        return unique_val
    
