# Super store prediction for sales
## Project Overview

This project aims to predict the future sales of a superstore using machine learning techniques. 
By leveraging historical data on sales, customer demographics, and various product categories, we strive to create a robust predictive model to help the store optimize inventory and improve decision-making

## Data set decription
- Row ID: Unique identifier for each row.

- Order ID: Unique identifier for each order.

- Order Date & Ship Date: Dates of order placement and shipment.

- Ship Mode: Shipping method chosen for the order (e.g., First Class, Standard).

- Customer ID & Customer Name: Customer information.

- Segment: Customer segment (e.g., Consumer, Corporate, Home Office).

- Country, City, State, Postal Code, Region: Geographic information related to the order.

- Product ID, Category, Sub-Category, Product Name: Product details.

- Sales: Sales revenue for each order line item

## Installation
git clone https://github.com/MissMukuru/Super_store_sales_prediction

cd superstore-sales-prediction

pip install -r requirements.txt

## Technologies used
- Python

- Pandas & NumPy

- Scikit-learn

- Matplotlib & Seaborn

- Jupyter Notebook

- Streamlit

- joblib

- SHAP

## Exploratory Data Analysis
Postal code was the only column with any missing values, I decided to drop it because it has very high cardinality and did not show any patterns in the Data

There are no duplicated values in the dataset

## Data cleaning
-I identified the outliers in the data set using interquatile ranges, defined the limits and removed the said outliers
![image](https://github.com/user-attachments/assets/8ba976d1-081a-43a8-9cf7-a5639c95aca8)


I converted the Order Date and Ship Date column to the required formarting for modelling and creating visualisations
![image](https://github.com/user-attachments/assets/71cb854b-49e0-4f01-9fd1-026f0e7f9154)

## Data visualisations
SALES TREND OVER TIME
![image](https://github.com/user-attachments/assets/9dec56be-43a6-47ed-a021-de70053f8ba2)

SALES BY DIFFRENT REGIONS
![image](https://github.com/user-attachments/assets/f25edcda-e910-4f43-bc12-298b2836e231)

HEATMAP FOR THE SALES PER CATEGORY
![image](https://github.com/user-attachments/assets/292a9987-103e-44f9-981e-7e75d3a64d7c)

Insights

1. In Technology generally has more sales than other categories in all the provided time period

2. In 2016 the Office supplies category did not do too well, but it subsequently got worse in the following years

3. Furniture maitained a steady decrease but in 2018 the sales dropped sharply.

4. Generally the sales seem to be declining each years

SALES IN DIFFERENT REGIONS
![image](https://github.com/user-attachments/assets/b9280326-04d9-49a5-82d6-b502df97353c)

Insights

1. Central had the least sales while South had the most sales
SALES BY CATEGORY AND SUB
![image](https://github.com/user-attachments/assets/78269e2d-b5d8-41ce-ab7f-a1a99a4bd5fa)

Insights

1. Furniture seems do be doing better than both Office supplies and Technology
2. Technology seems to be doing better than office supplies
3. Chairs have the highest sales while furnishings has the lowest under Furniture category
4. Storage has the highest sales while paper has the lowest sales in Office supplies category
5. Phones have the highest sales while copiers has the lowest sales in the Technology category
THE TOP 10 PRODUCTS BY SALES

SALES BY SEGMENT
![image](https://github.com/user-attachments/assets/1296b996-e512-4598-ab6a-2eee4ff5ab69)

Insights

1. The Consumer segment has the highest sales while the home office has the lowest
SALES BY SHIP MODE
![image](https://github.com/user-attachments/assets/082705dc-41e3-4960-ac55-79714723efdb)

Insights

The standard class has the highest sales while same day mode has the lowest
TOP CUSTOMERS
![image](https://github.com/user-attachments/assets/5a13a59a-0543-4653-b074-9d9313f75f58)

## Feature engineering
I split the Order Date and Ship Dates column into year, month, day and day of the week to make it easier for the model to recognize the patterns in the data.

modelling After spliting the data into train and test splits I used the Linear Regression Model to predict the sales

## Accuracy metrics
I used the Mean Squared Error, Mean Absolute Error and R-Squared to evaluate the Accuracy of my model.

## Model feature Importance
Here I used the SHAP package to gauge what feature are most influential in my model

![image](https://github.com/user-attachments/assets/95573526-9fb3-44d8-95de-59601094bcb6)


Insights

1. Region and state has very hig impact on the sales as the model has ranked variations of each higher
2. Category comes in a close second
3. Sub - Category also plays an important role in my models decisions

Saved my model for deployment using joblib

## DEPLOYMENT
I used streamlit to deploy my model

While There I added the navigation panel to show

1. The homepage
2. The Data information and EDA
3. Data visualisations
4. The prediction using the ML model

To run the model on your local machine

run as follows

streamlit run sales_prediction_model_pred.py
