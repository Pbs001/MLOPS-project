from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/puttu/Downloads/MLOPs-main/')
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__) # Entry point to all

app=application

## Route for a home page

@app.route('/') # render to index.html to search in templates folder
def index():
    return render_template('index.html') # to go to home page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=request.form.get('age'),
            marital=request.form.get('marital'),
            Personal_loan=request.form.get('personalLoan'),
            housing_loan=request.form.get('housingLoan'),
            ever_defaulted=request.form.get('everDefaulted')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        if(results[0]==1):
            results1 = 'Yes'
        else:
            results1 = 'No'
        return render_template('home.html',results=results1)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5656)        
