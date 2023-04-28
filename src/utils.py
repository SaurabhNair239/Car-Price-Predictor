import os
import sys
from sklearn.model_selection import RandomizedSearchCV
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException 
def save_object(file_path,obj_file):
    try:
        
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path,exist_ok=True)
        with open(file=file_path, mode="wb") as path:
            dill.dump(obj_file,path)
    except Exception as e:
        raise CustomException(e,sys)

def evaluation(X_train,y_train,X_test,y_test,model,parameters):
    try:
        r2_dict = {}
        for i in model.keys():
            model_obj = model[i]
            print(i)
            parameter = parameters[i]
            random_grid_model  = RandomizedSearchCV(model_obj,parameter,cv=5,n_jobs=-1,refit=False,scoring='r2')
            random_grid_model.fit(X_train,y_train)
            model_obj.set_params(**random_grid_model.best_params_)
            model_obj.fit(X_train,y_train)
            y_train_pred = model_obj.predict(X_train)
            y_test_pred = model_obj.predict(X_test)
    
            train_r2 = r2_score(y_train,y_train_pred)
            test_r2 = r2_score(y_test,y_test_pred)

            r2_dict[i] = [test_r2]
            print(r2_dict[i])
            r2_dict[i].append(model_obj)

        return r2_dict

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
        try:
            with open(file=file_path,mode="rb") as f_obj:
                return dill.load(f_obj)
        except Exception as e:
            raise CustomException(e,sys)