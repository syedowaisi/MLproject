import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerconfig:
    model_file_path=os.path.join('artifacts',"model.pkl")
    
class modeltrainer:
    def __init__(self):
        self.modeltrain_config=ModelTrainerconfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting train and test data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "adaboost regressor":AdaBoostRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report :dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                              models=models)
            #best model score
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score<=0.65:
                raise CustomException("no best model found")
            logging.info("best model is found for training and testing data")
            
            save_object(
                file_path=self.modeltrain_config.model_file_path,
                obj=best_model  
            )
            
            prediction=best_model.predict(x_test)
            r2_sq=r2_score(y_test,prediction)
            
            logging.info(f"accuracy score for best model {best_model} is:{r2_sq} ")
            
            return r2_sq
            
            
        except Exception  as e:
            raise CustomException(e,sys) 