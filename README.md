# Bitcoin Price Prediction - Machine Learning Model


## Bitcoin Price Prediction ML Model
In this project, I’ve developed a Machine Learning Model in Python to predict Bitcoin's future prices based on past price movements. This project involves data preparation, splitting, scaling, Sequential Machine Learning model building using TensorFlow (Keras), model training, testing, and building a web application on Streamlit that visualizes Bitcoin price predictions for the next five days. The data used for training the model is obtained from Yahoo Finance, covering the period from January 1, 2015, to December 31, 2023, with a focus on the closing prices.

## Data Preparation
+ Downloading Bitcoin base price data from Yahoo Finance.
+ Preparing a line chart based on the obtained data.
+ Slicing the data to focus on closing prices.
## Data Splitting & Scaling
+ Splitting the data into training and test datasets (due to BTC price fluctuation, I chose to take most of the data into the part of the training dataset so that our model can identify the data patterns with the time series data)
+ Scaling the data using MinMaxScaler to bring it within a specific range (0 to 1).
## Model Building
+ Building a Sequential Machine Learning model using LSTM layers with 50 nodes, dropout layers, and a dense output layer with one unit.
## Model Training & Testing
+ Training the model with the training dataset on the scaled transformed data.
+ Testing the model with the test dataset on the scaled transformed data.
## Model Prediction
+ Inverse transforming scaled data to go back to actual BTC prices
+ Predicting future Bitcoin prices for the next five days based on the trained model.
## Web Application Development
+ Saving the trained model as .kears filetype
+ Creating a Streamlit web application (BTC_app.py) for visualization.
## Web Application Execution
+ Running the Streamlit web application to display the line chart comparing predicted and original Bitcoin prices.


## Contributing
Contributions to enhance this analysis are welcome! 
Feel free to fork the repository and create a new branch for your modifications.

# Disclaimer
This analysis is based on personal data for a Python project ONLY and should **NOT** be construed as financial advice; it is solely intended for educational and informational purposes.
