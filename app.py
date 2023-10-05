import numpy as np
import pickle
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__,template_folder="templates")
model = pickle.load(open(r"D:\ml project\CO2.pickle",'rb'))
#column= pickle.load(open(r"C:\Users\yuvar\OneDrive\Desktop\Genetic classification\column G.pkl",'rb'))
print("loaded")
@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/pre')#redering to about page
def About():
   return render_template('index1.html')
# @app.route('/Prediction')#rendering to Prediction page
# def Prediction():
#    return render_template('result.html')
@app.route('/predict',methods=["POST","GET"])# route to showinput_feature=[float(x) for x in request.form.values() ] the predictions in a web UI
def prediction():
    cn=(request.form["cn"])
    cc=(request.form["cc"])
    ind=(request.form["in"])
    y=int(request.form["y"])

    # x = [float(x) for x in request.form.values()]
    x=[cn,cc,ind,y]
    x=np.array(x)


    input_data = {
            'country_name': [cn],
            'country_code': [cc],
            'indicator_name': [ind],
            'year': [y]
        }
    model_columns=['CountryName','CountryCode','IndicatorName','Year']
    input_df = pd.DataFrame(input_data)
    input_df = pd.get_dummies(input_df, columns=['country_name', 'country_code', 'indicator_name','year'])
    model_input = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(model_input)
    # prediction=model.predict([x])
   
    #if (prediction[0]==0):
    return render_template("result.html",predict=prediction[0])
    #else:
    #return render_template("result.html",predict=prediction[0])
     
if __name__=="__main__":
   app.run(debug = True,port = 1234)
   
