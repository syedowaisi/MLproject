import sys 
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import datatransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerconfig
from src.components.model_trainer import modeltrainer
@dataclass
class dataingestionconfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")
    
class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()
        
    def initiate_dataingestion(self):
        logging.info("entered the data ingestion method")
        
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split is initiated")
            
            train_test,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_test.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the dataset is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys) 
        
if __name__=="__main__":
    obj=dataingestion()
    train_data,test_data=obj.initiate_dataingestion()
    
    datatransform=datatransformation()
    train_arr,test_arr,_=datatransform.Initiate_data_transform(train_data,test_data)
    
    Modeltrainer=modeltrainer()
    r2score=Modeltrainer.initiate_model_trainer(train_arr,test_arr)
    
    print(f"model accuracy score is: {r2score}")