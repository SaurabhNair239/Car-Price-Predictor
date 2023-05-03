import pickle
import pandas as pd
from flask import Flask, render_template, request
import datetime
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipline.predict_pipeline import CustomizeData,PredictionPipeline




app = Flask(__name__)
@app.route("/",method=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict",method=["POST"])
def prediction():
    try:

     if request.method == 'POST':

          manufaturer = request.form['manufacturer']
          prod_year = int(request.form['Prod_year'])
          category = request.form['Category']
          leather_int = request.form['Leather_interior']
          fuel_type = request.form['Fuel_type']
          engine_vol = int(request.form['Engine_vol'])
          mileage = int(request.form['Mileage'])
          cylinder = float(request.form['Cylinders'])
          gear_box = request.form['Gear_box_type']
          turbos = request.form['Turbos']
          years_used = datetime.date.today().year - prod_year
          Customize_data_obj = CustomizeData(manufaturer, prod_year,category,leather_int,fuel_type,engine_vol,mileage,cylinder,gear_box,turbos,years_used)
          features = Customize_data_obj.get_data_as_frame()
          prediction_obj = PredictionPipeline()
          output = prediction_obj.predict(feature=features)

    except Exception as e:
         raise CustomException(e,sys) 