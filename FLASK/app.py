#loading the libraries
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
import os

#initialising the flask
app = Flask(__name__)



@app.route('/')
def f1():
    return render_template("home.html")

@app.route('/prediction',methods = ['post'])
def f2():
    if request.method == 'POST':
        results = request.form
        response = {}
        dic = {}

        for key,value in results.items():
            dic[key] = [value]

        df = pd.DataFrame(dic)
        df =  df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

        df.Gender.replace(['Male','Female'],[0,1],inplace=True)
        df.Married.replace(['No','Yes'],[0,1],inplace=True)
        df.Dependents = df.Dependents.str.replace("+","")
        df.Education.replace(['Graduate','Not Graduate'],[0,1],inplace=True)
        df.Self_Employed.replace(['No','Yes'],[0,1],inplace=True)
        df.Property_Area.replace(['Urban','Rural', 'Semiurban'],[0,1,2],inplace=True)

        int_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','ApplicantIncome']
        float_cols = ['CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

        for col in int_cols:
            df[col] = df[col].astype('int')
        for col in float_cols:
            df[col] = df[col].astype('float')

        x = df

        scaler = pickle.load(open('scaler.pkl','rb'))
        xgb = pickle.load(open('xgb.pkl','rb'))

        x = pd.DataFrame(scaler.transform(x),columns=scaler.get_feature_names_out())

        print(x)
        y_p = xgb.predict(x)
        if(y_p):
            response['result'] = "Congratulations! You may get approval for personal loan..."
            response['result_type'] = "positive"
        else:
            response['result'] = "Oops! you may not be eligible to get approval for personal loan..."
            response['result_type'] = "negative"

        return render_template('prediction.html',response=response)


if __name__ == '__main__':
    app.run(debug=True)