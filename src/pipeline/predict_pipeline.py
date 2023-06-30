import sys
import os
sys.path.append('/Users/puttu/Downloads/mlops12345/')
import pandas as pd
from src.exceptions import CustomException
from src.util import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        age: int,
        marital: str,
        Personal_loan: str,
        housing_loan: str,
        ever_defaulted: str):

        self.age = age

        self.marital = marital

        self.Personal_loan = Personal_loan

        self.housing_loan = housing_loan

        self.ever_defaulted = ever_defaulted

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "marital": [self.marital],
                "Personal_loan": [self.Personal_loan],
                "housing_loan": [self.housing_loan],
                "ever_defaulted": [self.ever_defaulted],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)