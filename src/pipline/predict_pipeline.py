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

    def __init__(self,manufaturer:str, prod_year:int, category:str, leather_int:str, fuel_type :str, engine_vol :int,
       mileage :int, cylinder :float, gear_box :str, turbos :str, years_used :int):
        
        
        self.Manufacturer = manufaturer
        self.Prod_year = prod_year
        self.Category = category
        self.Leather_int = leather_int
        self.Fuel_type = fuel_type
        self.Engine_vol= engine_vol
        self.Mileage = mileage
        self.Cylinder = cylinder
        self.Gear_box_type = gear_box
        self.Has_Turbo = turbos
        self.Years_used = years_used

    def get_data_as_frame(self):

        try:
            data_dict = {
                "Manufacturer":self.Manufacturer,
                "Prod_year":self.Prod_year,
                "Category":self.Category,
                "Leather_int":self.Leather_int,
                "Fuel_type":self.Fuel_type,
                "Engine_vol":self.Engine_vol,
                "Mileage":self.Mileage,
                "Cylinder":self.Cylinder,
                "Gear_box_type":self.Gear_box_type,
                "Has_Turbo":self.Has_Turbo,
                "Years_used":self.Years_used
                }

            data = pd.DataFrame(data=data_dict,index=[0])
            return data
        except Exception as e:
            raise CustomException(e,sys)


        

     