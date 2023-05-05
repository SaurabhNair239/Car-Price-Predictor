# Default of Credit Card Clients Dataset
It is an machine learning regression based model which is helpful in predicting the price of cars using different useful inputs.

### Workflow 
![ML Workflow](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/workflow.jpg)

Dataset: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Application link: http://18.193.67.220:8080/

Pyspark Data Analysis: https://colab.research.google.com/drive/1iEGi036Bk1hfBJRe2HWy4txRz92mBmmp?usp=sharing

## Data Analysis

###  Data Analysis using PySpark Commands 

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

![Average Price category](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/figure(3).png)

Car production in last 10 years

![10 years car production](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/figure(4).png)

Cars manufactured by KIA in the production year 2010

> df.select('*').filter("Manufacturer == 'KIA'").filter("Prod_year == '2010'").sort("Price",ascending=True).show()

Average Mileage of a every model manufactured by different companies

> df.groupBy("Manufacturer","Model").agg(avg("Mileage_km").alias("Average Mileage")).orderBy("Manufacturer").show(25)

Total sales in last 10 years

![10 years car production](https://github.com/SaurabhNair239/Car-Price-predictor/blob/main/images/figure(7).png)


** Note you can find data visualisation using plotly and EDA code in [EDA VISUALISATION](https://github.com/SaurabhNair239/Credit-card-default-predictor/blob/main/notebook/EDA.ipynb) file

## Deployment using AWS ECR and EC2 instance.

Contineous deployment can be done using the [main.yaml](https://github.com/SaurabhNair239/Car-Price-Predictor/blob/main/.github/workflows/main.yaml) file 

Before doing one must create AWS Elastic Container registry and user with required privilages as mentioned below:

> AmazonEC2FullAccess
> AmazonEC2ContainerRegistryFullAccess

After completing this process store the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as a github action secret key

Once the registry is being setup one must create a AWS ECS cluster and task defination as per the requirements

Final step will be to setup the EC2 instance and update the security requirments where you must specify the inbound and outbound TCP for IPv4 and IPv6.

Detailed explation of the deployment will be published soon.

## Next step:
* Improvement in the Frontend optimization.

