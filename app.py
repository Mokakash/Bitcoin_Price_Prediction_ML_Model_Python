import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import tensorflow as tf

#load Model 
model = load_model('https://github.com/Mokakash/Bitcoin_Price_Prediction_ML_Model_Python/blob/main/Bitcoin_Price_prediction_Model.keras')

st.header('Bitcoin Price Prediction ML Model')
st.subheader('Bitcoin Price Data [01/01/2015 - 12/31/2023]')
data = pd.DataFrame(yf.download('BTC-USD','2015-01-01','2023-12-31'))
st.write(data)

st.subheader('Bitcoin Price Chart')
data.drop(columns = ['Open','High','Low','Adj Close','Volume'], inplace=True)
st.line_chart(data)

train_data = data[:-100]
test_data = data[-200:]

scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i,0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0],x.shape[1],1))

st.subheader('BTC Predicted vs Original Price')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1,1)
ys = scaler.inverse_transform(y.reshape(-1,1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)
st.subheader('BTC Predicted vs Original Price Chart ')
st.line_chart(chart_data)

m = y
z= []
future_days = 6
for i in range(base_days, len(m)+future_days):
    m = m.reshape(-1,1)
    inter = [m[-base_days:,0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
    pred = model.predict(inter)
    m = np.append(m ,pred)
    z = np.append(z, pred)
st.subheader('BTC Predicted Future Price - 5 Days')
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1,1))
st.line_chart(z)