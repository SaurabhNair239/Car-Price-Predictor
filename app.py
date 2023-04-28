from src.logger import logging
from src.exception import CustomException
import sys
import streamlit as slt
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipline.predict_pipeline import CustomizeData,PredictionPipeline
slt.set_page_config(
     page_title='Credit Card Defaulter'
)

slt.title("Credit card default payment predictor")
slt.header("Predicts whether the customer will take the default payment option for next month")

LIMIT_BAL =  slt.number_input("Balance Limit",max_value=100000,min_value =10000,step=2000)

SEX = slt.selectbox("Gender",("Male","Female"))

MARRIAGE = slt.selectbox("Marital Status",("Married","Single","Other"))

EDUCATION = slt.selectbox("Education",("School Graduate","High School","University","Other"))

AGE = slt.slider("Age",max_value=60,min_value =21)

slt.write ("Please select the month due. Scales are as mentioned below and they are same for every month")

slt.write ("-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... , 8=payment delay for 8 month, 9=payment delay for 9 month or more")

slt.write()
PAY_0 = slt.selectbox("Payment status of September. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

PAY_1 = slt.selectbox("Payment status of August. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

PAY_2 = slt.selectbox("Payment status of July. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

PAY_3 = slt.selectbox("Payment status of June. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

PAY_4 = slt.selectbox("Payment status of May. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

PAY_5 = slt.selectbox("Payment status of April. How many months are due ?",(-1,1,2,3,4,5,6,7,8,9))

BILL_AMT_1 = slt.number_input("Bill amount of September",min_value=0,max_value=100000,step=500)

BILL_AMT_2 = slt.number_input("Bill amount of August",min_value=0,max_value=100000,step=500)

BILL_AMT_3 = slt.number_input("Bill amount of July",min_value=0,max_value=100000,step=500)

BILL_AMT_4 = slt.number_input("Bill amount of June",min_value=0,max_value=100000,step=500)

BILL_AMT_5 = slt.number_input("Bill amount of May",min_value=0,max_value=100000,step=500)

BILL_AMT_6 = slt.number_input("Bill amount of April",min_value=0,max_value=100000,step=500)

PAY_AMT1 = slt.number_input("Amount payment of before month in September",min_value=0,max_value=100000,step=500)

PAY_AMT2 = slt.number_input("Amount payment of before month in August",min_value=0,max_value=100000,step=500)

PAY_AMT3 = slt.number_input("Amount payment of before month in July",min_value=0,max_value=100000,step=500)

PAY_AMT4 = slt.number_input("Amount payment of before month in June",min_value=0,max_value=100000,step=500)

PAY_AMT5 = slt.number_input("Amount payment of before month in May",min_value=0,max_value=100000,step=500)

PAY_AMT6 = slt.number_input("Amount payment of before month in April",min_value=0,max_value=100000,step=500)

result = slt.button("Predict")

if result:

     slt.write("Your result will be soon displayed. Predicting..")
     slt.spinner()
     with slt.spinner(text="Work in Progress..."):
          Customize_data_obj = CustomizeData(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0,PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, BILL_AMT_1, BILL_AMT_2,BILL_AMT_3, BILL_AMT_4, BILL_AMT_5, BILL_AMT_6, PAY_AMT1,PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6)

          features = Customize_data_obj.get_data_as_frame()


          prediction_obj = PredictionPipeline()

          output = prediction_obj.predict(feature=features)
          
          if output[0] == 1:
             slt.write("Default Payment Next Month")
          else:     
             slt.write("Not a default payment next month")