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
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Random Forest":RandomForestRegressor(),
            }
            
            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'], 
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            model_report :dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                              models=models,param=params)
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
            
            logging.info(f"now after hyperparameter tuning the best model is {best_model} with accuracy score:{r2_sq}")
            
            return r2_sq
            
            
        except Exception  as e:
            raise CustomException(e,sys) 