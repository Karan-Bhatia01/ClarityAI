import io
import sys
import pandas as pd
import gridfs
from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        try:
            client   = MongoClient("mongodb://localhost:27017/")
            self.db  = client["clarityAI_database"]
            self.fs  = gridfs.GridFS(self.db)
            logging.info("MongoDB GridFS connection established")
        except Exception as e:
            raise CustomException(e, sys)

    def store_file(self, file):
        """Store uploaded file in MongoDB GridFS."""
        try:
            self.fs.put(
                file.read(),
                filename=file.filename,
                content_type=file.content_type,
            )
            logging.info("File '%s' stored in GridFS.", file.filename)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _read_csv_robust(file_bytes: bytes) -> pd.DataFrame:
        """
        Try to parse CSV bytes with progressively more lenient settings.
        Handles: encoding issues, inconsistent column counts, bad lines.
        """
        encodings = ["utf-8", "latin1", "utf-8-sig", "cp1252"]

        for encoding in encodings:
            for sep in [",", ";", "\t", "|"]:
                for bad_lines in ["skip", "warn"]:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(file_bytes),
                            encoding=encoding,
                            sep=sep,
                            on_bad_lines=bad_lines,
                            engine="python",
                        )
                        # Must have at least 1 column and 1 row
                        if df.shape[1] >= 1 and df.shape[0] >= 1:
                            logging.info(
                                "CSV parsed — encoding=%s sep=%r shape=%s",
                                encoding, sep, df.shape,
                            )
                            return df
                    except Exception:
                        continue

        raise ValueError("Could not parse the uploaded file as a valid CSV.")

    def get_preview(self):
        """
        Reads the latest uploaded file and returns first 5 rows + columns.
        Returns (None, None) if no file exists.
        """
        try:
            latest = self.fs.find().sort("uploadDate", -1).limit(1)
            for file in latest:
                file_bytes = file.read()
                df = self._read_csv_robust(file_bytes)
                return df.head().values.tolist(), df.columns.tolist()
            return None, None
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_filenames(self):
        """Return sorted list of CSV filenames only — excludes .pkl and other non-CSV files."""
        all_files = self.db.fs.files.distinct("filename")
        return sorted(f for f in all_files if f.lower().endswith(".csv"))