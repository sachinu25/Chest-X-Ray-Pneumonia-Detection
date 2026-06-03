import os
import sys

from xray.constant.training_pipeline import *
from xray.entity.artifacts_entity import DataIngestionArtifact
from xray.entity.config_entity import DataIngestionConfig
from xray.exception import XRayException
from xray.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config


    def get_data_from_s3(self) -> None:
        try:
            import glob, zipfile
            logging.info("Checking for local dataset zip file instead of S3")
            
            data_path = self.data_ingestion_config.data_path
            
            if not os.path.exists(data_path):
                zip_files = glob.glob(os.path.join(os.getcwd(), "*.zip"))
                if zip_files:
                    zip_file = zip_files[0]
                    logging.info(f"Extracting {zip_file} to {os.getcwd()}")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(os.getcwd())
                else:
                    logging.warning("No local dataset zip file found. Assuming data folder already exists.")
            else:
                logging.info(f"Data directory {data_path} already exists. Skipping extraction.")
                
            logging.info("Data extraction/validation complete")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(
            "Entered the initiate_data_ingestion method of Data ingestion class"
        )

        try:
            self.get_data_from_s3()

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info(
                "Exited the initiate_data_ingestion method of Data ingestion class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)
