import sys
import numpy as np
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
import os
from src.utils import save_object
from sklearn.base import BaseEstimator,TransformerMixin
import datetime

@dataclass
class DataTranformationSetting:
    preprocess_file_path = os.path.join("artifacts","preprocessor_pipeline.pkl")


class DataTranformation:
    def __init__(self):
        self.data_tranform_setting = DataTranformationSetting()

    def get_data_tranformer(self):
        try:
            numeric_cols = ["Prod_year","Engine_volume","Mileage","Cylinders","Years_used"]
            outlier_cols = ["Engine_volume","Mileage"]
            categorical_cols = ["Engine_volume","Mileage","Manufacturer","Category","Leather_interior","Fuel_type","Gear_box_type","Has_Turbo","Years_used"]
            num_pipeline = Pipeline(
                 steps=[
                ("Standard scaler",StandardScaler())
                 ]   
                ) 
                
            logging.info("Pipeline completed for numerical data")
            categorical_pipeline = Pipeline(
                steps=[
                ("OnehotEncoder",OneHotEncoder(sparse=False,handle_unknown='ignore')),
                ]
            )
            logging.info("Pipeline completed categorical pipeline")


            numerical_oulier_pipeline = Pipeline(
                steps=[
                ("Onlier removal",OutlierTreatment()),
                ("Simple Imputer",SimpleImputer(strategy="median"))
                ]
            )

            logging.info("Outlier removal treatment done")

            preprocess = ColumnTransformer(
                [
                ('Oulier removal Pipeline',numerical_oulier_pipeline,outlier_cols),
                ('num_pipeline',num_pipeline,numeric_cols),
                ('cat_pipeline',categorical_pipeline,categorical_cols)
                ]
            )

            logging.info("Combined both the pipelines")
            return preprocess

        except Exception as e:
            raise CustomException(e,sys)



    def start_data_tranformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Loaded Train and Test data")

            logging.info("getting preprocessing steps")

            preprocess_obj = self.get_data_tranformer()

            target_column = "Price"

            input_feature_train = train_data.drop([target_column,"ID","Levy",'Drive wheels', 'Doors', 'Wheel', 'Color',"Model",'Airbags'],axis=1)
            input_feature_test = test_data.drop([target_column,"ID","Levy",'Drive wheels', 'Doors', 'Wheel', 'Color',"Model",'Airbags'],axis=1)

            logging.info("Renaming Columns")
            input_feature_train.columns = ['Manufacturer','Prod_year','Category','Leather_interior','Fuel_type','Engine_volume','Mileage','Cylinders','Gear_box_type','Has_Turbo','Years_used']
            input_feature_test.columns = ['Manufacturer','Prod_year','Category','Leather_interior','Fuel_type','Engine_volume','Mileage','Cylinders','Gear_box_type','Has_Turbo','Years_used']


            input_train_delete_rows = input_feature_train[input_feature_train["Manufacturer"].isin(input_feature_train["Manufacturer"].value_counts().tail(24).index.to_list())].index.to_list()
            input_test_delete_rows = input_feature_test[input_feature_test["Manufacturer"].isin(input_feature_train["Manufacturer"].value_counts().tail(24).index.to_list())].index.to_list()

            input_feature_train.drop(input_train_delete_rows,inplace=True)
            input_feature_train = input_feature_train.reset_index(drop=True)

            input_feature_test.drop(input_test_delete_rows,inplace=True)
            input_feature_test = input_feature_test.reset_index(drop=True)
            

            input_feature_train["Mileage"] = input_feature_train["Mileage"].apply(lambda x: x.replace("km",""))
            input_feature_train["Has_Turbo"] = np.where(input_feature_train["Mileage"].str.contains("Turbo"),"Yes","No")
            input_feature_train["Engine volume"] = input_feature_train["Engine volume"].apply(lambda x: x.replace("Turbo",""))
            input_feature_train["Mileage"] = input_feature_train["Mileage"].astype(int)
            input_feature_train["Engine volume"] = input_feature_train["Engine volume"].astype(float)

            input_feature_test["Mileage"] = input_feature_test["Mileage"].apply(lambda x: x.replace("km",""))
            input_feature_test["Has_Turbo"] = np.where(input_feature_test["Mileage"].str.contains("Turbo"),"Yes","No")
            input_feature_test["Engine volume"] = input_feature_test["Engine volume"].apply(lambda x: x.replace("Turbo",""))
            input_feature_test["Mileage"] = input_feature_test["Mileage"].astype(int)
            input_feature_test["Engine volume"] = input_feature_test["Engine volume"].astype(float)

            input_feature_train["Years_used"] = datetime.date.today().year - input_feature_train["Prod. year"]
            input_feature_test["Years_used"] = datetime.date.today().year - input_feature_test["Prod. year"]

            logging.info("Data Cleaned")

            train_target_feature = train_data[target_column]
            test_target_feature = test_data[target_column]
            
            logging.info("Preprocessing train data")
            train_input_data_preprocessed = preprocess_obj.fit_transform(input_feature_train)

            logging.info("Preprocessing test data")
            test_input_data_preprocessed = preprocess_obj.transform(input_feature_test)

            logging.info("Combining target and input features")

            train_data = np.c_[train_input_data_preprocessed,np.array(train_target_feature)]
            test_data = np.c_[test_input_data_preprocessed,np.array(test_target_feature)]

            logging.info("Preprocessing complete")

            logging.info("Saving Preprocessing pipeline Pickle file")
            save_object(
                    file_path= self.data_tranform_setting.preprocess_file_path,
                    obj_file = preprocess_obj
            )      
            logging.info("Saved Preprocessing pipeline Pickle file")      
            return train_data,test_data,self.data_tranform_setting.preprocess_file_path

        except Exception as e:
            raise CustomException(e,sys)
        

class OutlierTreatment(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        self.lower_bound = []
        self.upper_bound = []

    def outlier_ub_lb(self,X):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3-q1
        self.lower_bound.append(q1 - (1.5*iqr))
        self.upper_bound.append(q3 + (1.5*iqr))

    

    def outlier_data_transformer(self,X):
        X.apply(self.outlier_ub_lb)
        for i in range(X.shape[1]):
            copy_data = X.iloc[:,i].copy()
            copy_data[(copy_data < self.lower_bound[i]) & (copy_data > self.upper_bound[i])] = np.nan
            X.iloc[:,i]=copy_data
        return X

               
    def fit_transform(self,X,y=None):
        
        try:
            data_transformed = self.outlier_data_transformer(X)
            return data_transformed
        except Exception as e:
            raise CustomException(e,sys)
        
    def transform(self,X,y=None):
        try:
            for i in range(X.shape[1]):
                copy_data = X.iloc[:,i].copy()
                copy_data[(copy_data < self.lower_bound[i]) & (copy_data > self.upper_bound[i])] = np.nan
                X.iloc[:,i]=copy_data    
            return X
        except Exception as e:
            raise CustomException(e,sys)
        
    def fit(self,X,y=None):
        try:
            self.outlier_data_transformer(X)
            return self
        except Exception as e:
            raise CustomException(e,sys)