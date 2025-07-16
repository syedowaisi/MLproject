import os
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") 
    
class datatransformation:
    def __init__(self):
        self.data_transform_config=DataTransformationConfig()
        
    def get_data_transform_object(self):
        '''
        this function is responsible for data transformation 
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=["gender",
                                "race_ethnicity",
                                "parental_level_of_education",
                                "lunch",
                                "test_preparation_course"
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standard scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def Initiate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("reading of train and test data")
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transform_object()
            
            target_column="math_score"
            numerical_col=["writing_score","reading_score"]
            
            input_train_feature=train_df.drop(columns=[target_column],axis=1)
            target_train_feature=train_df[target_column]
            
            input_test_feature=test_df.drop(columns=[target_column],axis=1)
            target_test_feature=test_df[target_column]
            
            logging.info(f"applying preprocessing object on training and testing dataframe") 
            
            input_train_feature_transform=preprocessing_obj.fit_transform(input_train_feature)
            input_test_feature_transform=preprocessing_obj.transform(input_test_feature)
            
            train_arr=np.c_[
                input_train_feature_transform,np.array(target_train_feature)
            ]
            test_arr=np.c_[
                input_test_feature_transform,np.array(target_test_feature)
            ]
            
            logging.info("saved preprocessing object")
            
            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
            