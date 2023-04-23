import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as dr
import datetime as dt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from io import BytesIO
from datetime import date, timedelta
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from minio import Minio
import pickle 
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)


load_dotenv()
minio_client = None
BUCKET = os.getenv("S3_BUCKET")
minio_client = Minio('localhost:9002', os.getenv(
        "S3_ACCESS_KEY"), os.getenv("S3_SECRET_KEY"), secure=False)

def get_lastest_pkl(filename):
    """Get the lastest pickle file's path from Minio"""
    list_items = [
        item.object_name
        for item in minio_client.list_objects(BUCKET, "bitcoin_artifacts", recursive=True)
        if item.object_name.endswith(filename)
    ]
    last_modified = [
        minio_client.stat_object(BUCKET, path).last_modified for path in list_items
    ]
    path_df = pd.DataFrame(
        list(zip(list_items, last_modified)), columns=["path", "last_modified"]
    )
    pkl_path = path_df.loc[
        path_df.last_modified == path_df.last_modified.max(), "path"
    ].values[0]
    return pkl_path

st.title("Cryptocurrency Trend Prediction")

df = yf.download(tickers='BTC-USD', period = '60d', interval = '15m')

st.subheader('BTC-USD for the last 60 days')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_array = scaler.fit_transform(train) 

path_model = get_lastest_pkl("model.pkl")
obj_model = minio_client.get_object(BUCKET, path_model)
logging.info("loading data from Minio : %s", path_model)
model = pickle.load(obj_model)

past_100_days = train.tail(100)
final_df = past_100_days.append(test, ignore_index=True)
input_data = scaler.fit_transform(final_df) 


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)