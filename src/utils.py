import os
import sys
import numpy as np 
import pandas as pd 
import dill 
import pickle
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV 

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True) 
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)  
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            parameter=param[list(models.keys())[i]]  
            
            gridsearch=GridSearchCV(model,parameter,cv=3)
            gridsearch.fit(x_train,y_train) 
            model.set_params(**gridsearch.best_params_)
            model.fit(x_train,y_train)
            
            y_pred_train=model.predict(x_train)
            y_pred_test=model.predict(x_test)
            
            train_model_score=r2_score(y_pred_train,y_train)  
            
            test_model_score=r2_score(y_pred_test,y_test)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report 
    
    except Exception as e:
        raise CustomException(e,sys)
    
            
            