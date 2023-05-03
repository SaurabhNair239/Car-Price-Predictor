from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
import numpy as np
from src.utils import evaluation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os
from src.utils import save_object

@dataclass
class model_training_settings:
    train_model_path = os.path.join("artifacts","model.pkl")


class model_trainer:
    def __init__(self):
        self.model_path = model_training_settings()

    def training_model(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Splitting input and target variable")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )       
            models = {
                      "Linear Regression": LinearRegression(),           
                      "Random Forest": RandomForestRegressor(),
                      "Decision Tree": DecisionTreeRegressor(),
                      "XGB Regressor": XGBRegressor(),
                      "Ridge": Ridge(),
                      "Lasso": Lasso(), 
                      "KNNR": KNeighborsRegressor(),
                      "AdaBoost Regressor": AdaBoostRegressor(),
                      "CatBoost Regressor" : CatBoostRegressor()
            }     
            

            hyper_parameter = {
                "Linear Regression": {},
                "AdaBoost Regressor":{
                
                    'learning_rate': [0.1,0.01,0.5,0.001],
                    'n_estimators': [7,14,21,28,35]

                },

                "Random Forest":{
                    'criterion':["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    'max_depth': [10, 20],
                    'n_estimators': [200, 400]
                },
                
                "Decision Tree":{
                    'criterion':["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "max_depth" : [1,3,9],
                    'min_samples_split': [0.2,0.3,0.5],
              },

                "CatBoost Regressor": {
                    'depth': [4,5,6],
                    'learning_rate': [0.01,0.02,0.03,0.04],
                    'iterations': [300,400]
                },

                "XGB Regressor" :{
                     'learning_rate':[0.1,0.01,0.001],
                     'n_estimators': [4,6,10] 
                },

                "KNNR":{
                
                    'n_neighbors': list(range(1,15,2))
                },

                "Lasso":{
                    
                    "alpha": list(np.arange(0.1, 1.0, 0.3))
                },
    
                "Ridge":{
                
                    "alpha": list(np.arange(0.1, 1.0, 0.3))
                }
            }

            report = evaluation(X_train,y_train,X_test,y_test,model=models,parameters=hyper_parameter)


            report_r2 = [i[0] for i in report.values()]
            model_data = [i[1] for i in report.values()]
            best_model = model_data[report_r2.index(min(report_r2))]
            report_keys = [i for i in report.keys()]
            model_name = report_keys[report_r2.index(min(report_r2))]
            print("Best Model: ",model_name)
            logging.info("Found best model")

            save_object(file_path=self.model_path.train_model_path,obj_file=best_model)
            
            prediction = best_model.predict(X_test)

            acc = r2_score(y_test,prediction)

            return acc 
        except Exception as e:
            raise CustomException(e,sys)
