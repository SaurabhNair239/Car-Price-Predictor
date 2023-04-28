import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
        
    def predict(self,feature):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor_pipeline.pkl"
            model = load_object(file_path = model_path)
            preprocessor_model = load_object(file_path = preprocessor_path)

            data_transformed = preprocessor_model.transform(feature)
            predicted_val = model.predict(data_transformed)

            return predicted_val
        except Exception as e:
            raise CustomException(e,sys)    

class CustomizeData:

    def __init__(self,LIMIT_BAL:int, SEX:str, EDUCATION:str, MARRIAGE:str, AGE:int, PAY_0:int,
       PAY_1:int, PAY_2:int, PAY_3:int, PAY_4:int, PAY_5:int, BILL_AMT1:int, BILL_AMT2:int,
       BILL_AMT3:int, BILL_AMT4:int, BILL_AMT5:int, BILL_AMT6:int, PAY_AMT1:int,
       PAY_AMT2:int, PAY_AMT3:int, PAY_AMT4:int, PAY_AMT5:int, PAY_AMT6:int):
        
        
        self.limit = LIMIT_BAL
        self.gender = SEX
        self.education = EDUCATION
        self.marriage = MARRIAGE
        self.age = AGE
        self.pay0 = PAY_0
        self.pay1 = PAY_1
        self.pay2 = PAY_2
        self.pay3 = PAY_3
        self.pay4 = PAY_4
        self.pay5 = PAY_5

        self.bill1 = BILL_AMT1
        self.bill2 = BILL_AMT2
        self.bill3 = BILL_AMT3
        self.bill4 = BILL_AMT4
        self.bill5 = BILL_AMT5
        self.bill6 = BILL_AMT6

        self.payamt1 = PAY_AMT1
        self.payamt2 = PAY_AMT2
        self.payamt3 = PAY_AMT3
        self.payamt4 = PAY_AMT4
        self.payamt5 = PAY_AMT5
        self.payamt6 = PAY_AMT6



        if self.education == "School Graduate":
            self.education = "School_g"
        elif self.education == "High School":
            self.education = "High_s"
        elif self.education == "University":
            self.education = "University"
        else:
            self.education = "Other"

    def get_data_as_frame(self):

        try:
            data_dict = {
                "LIMIT_BAL":self.limit,
                "SEX":self.gender,
                "EDUCATION":self.education,
                "MARRIAGE":self.marriage,
                "AGE":self.age,
                "PAY_0":self.pay0,
                "PAY_2":self.pay1,
                "PAY_3":self.pay2,
                "PAY_4":self.pay3,
                "PAY_5":self.pay4,
                "PAY_6":self.pay5,
                "BILL_AMT1":self.bill1,
                "BILL_AMT2":self.bill2,
                "BILL_AMT3":self.bill3,
                "BILL_AMT4":self.bill4,
                "BILL_AMT5":self.bill5,
                "BILL_AMT6":self.bill6,
                "PAY_AMT1":self.payamt1,
                "PAY_AMT2":self.payamt2,
                "PAY_AMT3":self.payamt3,
                "PAY_AMT4":self.payamt4,
                "PAY_AMT5":self.payamt5,
                "PAY_AMT6":self.payamt6                
            }

            data = pd.DataFrame(data=data_dict,index=[0])
            return data
        except Exception as e:
            raise CustomException(e,sys)


        

     