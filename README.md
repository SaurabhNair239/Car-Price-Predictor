# Car Price Prediction
It is an machine learning regression based model which is helpful in predicting the price of cars using different useful inputs.

* 1. [Workflow](#Workflow)
* 2. [Data Analysis](#DataAnalysis)
	* 2.1. [ Data Analysis using PySpark Commands](#DataAnalysisusingPySparkCommands)
* 3. [Deployment using AWS ECR and EC2 instance.](#DeploymentusingAWSECRandEC2instance.)
	* 3.1. [Below mentioned steps need to be followed for deployment](#Belowmentionedstepsneedtobefollowedfordeployment)
* 4. [Next step:](#Nextstep:)

##  1. <a name='Workflow'></a>Workflow 
![ML Workflow](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/car_price_workflow.jpg)

Dataset: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Application link: http://18.193.67.220:8080/

Pyspark Data Analysis: https://colab.research.google.com/drive/1iEGi036Bk1hfBJRe2HWy4txRz92mBmmp?usp=sharing

##  2. <a name='DataAnalysis'></a>Data Analysis

###  2.1. <a name='DataAnalysisusingPySparkCommands'></a> Data Analysis using PySpark Commands 

Different Manufacturer order by alphabatic order

> df.select("Manufacturer").distinct().orderBy("Manufacturer").show(10)

Top 10 manufacturer by count

> df.groupBy("Manufacturer").count().orderBy("count",ascending=False).show()

![Top 10 Manufatures](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot.png)

Top 10 car models

> df.groupBy("Manufacturer","Model").count().orderBy("count",ascending=False).show(10)

![Top 10 car Model](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot(1).png)
 
Manufacturer with Leather Interior

> df.select("Manufacturer","Model","Price").filter("Leather_interior == 'Yes'").show()

![Manufacturer Leather Interior](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot(2).png)

Average Price of each Category 

> df.groupBy("Category").agg(avg("Price").alias("Average Price")).orderBy("Average Price",ascending=False).show(20)

![Average Price category](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot(3).png)

Car production in last 10 years

![10 years car production](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot(4).png)

Cars manufactured by KIA in the production year 2010

> df.select('*').filter("Manufacturer == 'KIA'").filter("Prod_year == '2010'").sort("Price",ascending=True).show()

Average Mileage of a every model manufactured by different companies

> df.groupBy("Manufacturer","Model").agg(avg("Mileage_km").alias("Average Mileage")).orderBy("Manufacturer").show(25)

Total sales in last 10 years

![10 years car production](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/newplot(7).png)


** Note you can find data visualisation using plotly and EDA code in [EDA VISUALISATION](https://github.com/SaurabhNair239/Credit-card-default-predictor/blob/main/notebook/EDA.ipynb) file

##  3. <a name='DeploymentusingAWSECRandEC2instance.'></a>Deployment using AWS ECR and EC2 instance.

Contineous deployment can be done using the [main.yaml](https://github.com/SaurabhNair239/Car-Price-Predictor/blob/main/.github/workflows/main.yaml) file 

###  3.1. <a name='Belowmentionedstepsneedtobefollowedfordeployment'></a>Below mentioned steps need to be followed for deployment

* Create AWS Elastic Container registry .

* Create user with below mentioned privilages:

    *  AmazonEC2FullAccess

    * AmazonEC2ContainerRegistryFullAccess

* Store AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in github action secret key.

* Create a AWS ECS cluster and task defination as per the requirements.

* Setup the EC2 instance

* Update the security requirments where you must specify the inbound and outbound TCP for IPv4 and IPv6.

***Detailed explation of the deployment will be published soon.***

##  4. <a name='Nextstep:'></a>Next step:

* Improvement in the Frontend optimization.

