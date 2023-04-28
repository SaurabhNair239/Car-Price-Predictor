import os
import sys
import boto3
from io import StringIO
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTranformation
# from src.components.data_transformation import DataTranformationSetting
from src.components.model_trainer import model_trainer
@dataclass
class DataIngestionPaths:
    train_path = os.path.join("artifacts","train.csv")
    test_path = os.path.join("artifacts","test.csv")
    raw_path = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_path = DataIngestionPaths()

    def start_ingestion_data(self):
        logging.info("Started Data ingestion method")
        try:
            s3_resource = boto3.client('s3', aws_access_key_id="AKIASPLFXLBWTP6LSWCQ",
                               aws_secret_access_key="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")
            
            obj = s3_resource.get_object(Bucket='car-price-bucket', Key="car_price_prediction.csv")

            data = obj['Body'].read().decode('utf-8')

            data = StringIO(data)

            data = pd.read_csv(data)    

            logging.info("Loaded data as DF")

            os.makedirs(os.path.dirname(self.ingestion_path.test_path),exist_ok=True)

            data.to_csv(self.ingestion_path.raw_path,index=False,header=True)

            logging.info("Initiated Train Test Split")

            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_path.train_path,index=False,header=True)

            test_set.to_csv(self.ingestion_path.test_path,index=False,header=True)

            logging.info("Completed Data Ingestion")

            return (
                self.ingestion_path.train_path,
                self.ingestion_path.test_path
            )
        except Exception as e :
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_injestion_obj = DataIngestion()
    train_path,test_path = data_injestion_obj.start_ingestion_data()
    # dd = pd.read_csv(train_path)
    # print(dd.info())
    # print(os.environ['server'],os.environ['db'],os.environ['user'])
    data_transform_obj = DataTranformation()
    train_data,test_data,path = data_transform_obj.start_data_tranformation(train_path,test_path)
    model_trainer_obj = model_trainer()
    r2_value = model_trainer_obj.training_model(train_arr=train_data,test_arr = test_data,preprocessor_path=path)
    print(r2_value)