import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import seaborn as sns  # Seaborn import as you mentioned

st.title("Sales Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio('Go to', ['Homepage', 'Data Information', 'Visualizations', 'Machine learning model'])

# Load the data
data = pd.read_csv("C:/Users/HPPC/Desktop/sales_prediction/data/super_store.csv")

# Homepage
if option == 'Homepage':
    st.title('Welcome to the Sales Prediction App!')
    st.write("Are you looking to gain insights into sales trends and make data-driven decisions?")
    
    st.write("You've come to the right place! This app is designed to help you explore and visualize sales data, allowing you to understand patterns, track performance, and make predictions about future sales.")
    st.write('What You Can Do Here:')
    st.write('Exploration: Dive into the data, view summary statistics, and check for any missing or duplicate entries.')
    st.write('Visualizations: Discover trends in sales over time, explore regional and segment-wise sales, and identify top products and customer segments.')
    st.write('Sales Prediction: Using a machine learning model, you can predict sales performance based on key variables like shipping mode, segment, and order date.')
    st.write("With this app, you'll be able to interact with the data and uncover valuable insights that can drive business success. Enjoy exploring")

elif option == 'Data Information':
    # Display dataset information
    st.write('Data Information this is the part of the app where you find interesting insights about the data')
    st.write('The first five rows of the dataset:', data.head())
    st.write('Summary statistics of the dataset:', data.describe())
    st.write("The total number of duplicated rows are:", data.duplicated().sum())
    st.write("The number of rows and columns in the data is:", data.shape)

    # Missing values checkbox
    if st.checkbox("Check the total number of missing values"):
        st.write(data.isnull().sum().sum())

elif option == 'Visualizations':
    # Data visualizations
    st.write('Data Visualizations')

    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')


    # Sales over time
    st.write("Sales over time")
    fig1 = plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x='Order Date', y='Sales')
    plt.title('Sales Trend')
    plt.xlabel('Order Date')
    plt.ylabel('Sales')
    st.pyplot(fig1)

    # Other visualizations continue here...
    st.write('Top 10 products by sales')
    Top_10_sales = data[['Product Name', 'Sales']].sort_values(by = 'Sales', ascending = False).head(10)
    fig2 = plt.figure(figsize=(10, 5))
    sns.barplot(data = Top_10_sales, x = 'Product Name', y = 'Sales')
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Product Name')
    plt.ylabel('Sales')
    plt.xticks(rotation = 90)
    st.pyplot(fig2)

    st.write('Sales by Segment')
    Segmented_sales = data.groupby('Segment')['Sales'].sum()
    fig3 = plt.figure(figsize=(10, 5))
    sns.barplot(x = Segmented_sales.index, y = Segmented_sales)
    plt.title('Sales by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Sales')
    st.pyplot(fig3)

    st.write('Sales by ship')
    Popular_shipmodes = data.groupby('Ship Mode')['Sales'].sum().sort_values(ascending = False).head(5)
    fig4 = plt.figure(figsize=(10, 5))
    sns.barplot(x = Popular_shipmodes.index, y = Popular_shipmodes)
    plt.title('Sales by Ship Mode')
    plt.xlabel('Ship Mode')
    st.pyplot(fig4)

    st.write('Top 10 customers')
    Sales_by_customer = data.groupby('Customer Name')['Sales'].sum().sort_values(ascending = False).head(10)
    fig5 = plt.figure(figsize=(10, 5))
    sns.barplot(x = Sales_by_customer.index, y = Sales_by_customer)
    plt.title('Sales by Customer')
    plt.xlabel('Customer Name')
    plt.xticks(rotation = 90)
    st.pyplot(fig5)

    st.write('Order quantity by order ID')
    Order_quantity = data.groupby('Order Date')['Order ID'].count()
    fig6 = plt.figure(figsize = (10, 10))
    sns.lineplot(data = Order_quantity.reset_index(), x = 'Order Date', y = 'Order ID')
    plt.title('Order Quantity by Order ID')
    plt.xlabel('Order ID')
    st.pyplot(fig6)

    st.write('Sales by different categories')
    fig7 = plt.figure(figsize=(10, 5))
    sns.barplot(data = data, x = 'Category', y = 'Sales')
    plt.title('Sales by different categories')
    plt.xlabel('Region')
    plt.ylabel('Sales')
    st.pyplot(fig7)

    st.write('heatmap showing the sales per caterogy in years')
    data['Order Year'] = data['Order Date'].dt.year
    heatmap_data = pd.pivot_table(data, values='Sales', index='Category', columns='Order Year')
    fig8 = plt.figure(figsize = (10, 5))
    sns.heatmap(data = heatmap_data, annot = True)
    plt.title('sales per category in years')
    plt.show()
    st.pyplot(fig8)

    st.write('Sales in different regions')
    fig9 = plt.figure(figsize=(10, 5))
    sns.barplot(data = data, x = 'Region', y = 'Sales')
    plt.title('Sales in Different Regions')
    plt.xlabel('Region')
    plt.ylabel('Sales')
    st.pyplot(fig9)

    st.write('Sales by category and sub-category')
    grouped_sales = data.groupby(['Category', 'Sub-Category'])['Sales'].sum()
    grouped_sales = grouped_sales.reset_index()
    fig10 = plt.figure(figsize=(10, 5))
    sns.barplot(data = grouped_sales, x = 'Category', y = 'Sales', hue = 'Sub-Category')
    plt.title('Sales by Category and Sub-Category')
    plt.xlabel('Category')
    plt.ylabel('Sales')
    plt.legend(title='Sub-Category')
    plt.show()
    st.pyplot(fig10)




elif option == 'Machine learning model':
    st.write("Welcome to the sales prediction")

    # Load the model
    model = joblib.load("C:/Users/HPPC/Desktop/sales_prediction/modelling/logistic_model.pkl")
    st.write("Model loaded successfully, go ahead and make predictions")

    # Input fields for user
    Ship_Mode = st.selectbox('Ship Mode', ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    Segment = st.selectbox('Segment', ['Consumer', 'Home Office', 'Corporate'])
    Region = st.selectbox('Region', ['South', 'West', 'Central', 'East'])
    Sub_Category = st.selectbox('Sub-Category', ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper'])
    Order_Date = st.date_input("Select Order Date", value=pd.Timestamp.now())

    # Extract date features
    Order_Year = Order_Date.year
    Order_Month = Order_Date.month
    Order_Day = Order_Date.day
    Order_Day_of_week = Order_Date.weekday()

    # Button for prediction
    if st.button("Predict"):
        # Input data preparation
        input_data = pd.DataFrame({
            'Ship_Mode': [Ship_Mode],
            'Segment': [Segment],
            'Region': [Region],
            'Sub-Category': [Sub_Category],
            'Order Year': [Order_Year],
            'Order Month': [Order_Month],
            'Order Day': [Order_Day],
            'Order Day of Week': [Order_Day_of_week]
        })

        # Mapping categorical fields
        input_data['Ship_Mode'] = input_data['Ship_Mode'].map({'Standard Class': 0, 'Second Class': 1, 'First Class': 2, 'Same Day': 3})
        input_data['Segment'] = input_data['Segment'].map({'Consumer': 0, 'Home Office': 1, 'Corporate': 2})
        input_data['Region'] = input_data['Region'].map({'South': 0, 'West': 1, 'Central': 2, 'East': 3})
        input_data['Sub-Category'] = input_data['Sub-Category'].map({'Bookcases': 0, 'Chairs': 1, 'Labels': 2, 'Tables': 3, 'Storage': 4, 'Furnishings': 5, 'Art': 6, 'Phones': 7, 'Binders': 8, 'Appliances': 9, 'Paper': 10})

        # Aligning input data columns with model
        missing_cols = [col for col in model.feature_names_in_ if col not in input_data.columns]
        for col in missing_cols:
            input_data[col] = 0

        input_data = input_data[model.feature_names_in_]

        # Prediction
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.write("Prediction: Sales are predicted to be high.")
        else:
            st.write("Prediction: Sales are predicted to be low.")
