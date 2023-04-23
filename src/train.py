# Raw Package
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from io import BytesIO
from datetime import date, timedelta
from datetime import datetime
import time
import os
import pickle
from dotenv import load_dotenv
from minio import Minio
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

def get_minio_credentials():
    """
    Returns the credentials required to connect to Minio.
    The credentials are read from environment variables using the `dotenv` library.
    """
    logging.info("Reading env variables to connect to Minio ...")
    load_dotenv()
    bucket_name = os.getenv("S3_BUCKET")
    access_key_id = os.getenv("S3_ACCESS_KEY")
    secret_access_key = os.getenv("S3_SECRET_KEY")
    minio_api_url = os.getenv("S3_URL")
    return bucket_name, access_key_id, secret_access_key, minio_api_url


def create_minio_client(minio_api_url, access_key_id, secret_access_key, bucket_name):
    """
    Creates a Minio client object and checks connection to the specified bucket.

    Args:
        minio_api_url (str): The URL of the Minio server.
        access_key_id (str): The access key ID for the Minio server.
        secret_access_key (str): The secret access key for the Minio server.
        bucket_name (str): The name of the bucket to check for connection.

    Returns:
        A `minio.Minio` object representing the Minio client.

    Raises:
        `minio.error.ResponseError` if the bucket is not reachable.
    """
    logging.info("Creating Minio Client ...")
    client = Minio(
        minio_api_url,
        access_key=access_key_id,
        secret_key=secret_access_key,
        secure=False
    )
    client.bucket_exists(bucket_name)
    logging.info("Minio object storage connected")

    return client

def parse_date():
    """
    Returns the current day, month, and year as integers.
    """
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    return day, month, year


def model_arch(x_train):
    model = Sequential()
    model.add(LSTM( units=50, activation='relu', return_sequences = True,
                input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM( units=60, activation='relu', return_sequences = True))
    model.add(Dropout(0.3))

    model.add(LSTM( units=80, activation='relu', return_sequences = True))
    model.add(Dropout(0.4))
            
    model.add(LSTM( units=120, activation='relu'))
    model.add(Dropout(0.5))
            
    model.add(Dense(units = 1))
    return model

def model_forecast(model, X, window_size):
    """Takes in numpy array, creates a windowed tensor 
    and predicts the following value on each window"""
    data = tf.data.Dataset.from_tensor_slices(X)
    data = data.window(window_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda w: w.batch(window_size))
    data = data.batch(32).prefetch(1)
    forecast = model.predict(data)
    return forecast

def df_from_bucket(minio_client, bucket, path):
    obj = minio_client.get_object(
        bucket,
        path,
    )
    out_df = pd.read_csv(obj, lineterminator='\n')
    return out_df

def main():
    load_dotenv()
    start_time = time.time()
    bucket_name, access_key_id, secret_access_key, minio_api_url = get_minio_credentials()
    minio_client = create_minio_client(minio_api_url, access_key_id, secret_access_key, bucket_name)

    day, month, year = parse_date()
    btc_path = f"bitcoin_data/{day}-{month}-{year}/bitcoin.csv"
    btc_data = df_from_bucket(minio_client, bucket_name, btc_path)
    train = pd.DataFrame(btc_data['Close'][0:int(len(btc_data)*0.7)])
    test = pd.DataFrame(btc_data['Close'][int(len(btc_data)*0.7):int(len(btc_data))])
    logging.info(f"Training data shape: {train.shape}")
    logging.info(f"Validation data shape: {test.shape}")
    scaler = MinMaxScaler(feature_range=(0,1))
    train_array = scaler.fit_transform(train) 
    x_train = []
    y_train = []

    for i in range(100, train_array.shape[0]):
        x_train.append(train_array[i-100:i])
        y_train.append(train_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    model = model_arch(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5)
    logging.info("Saving model Artifacts ...")
    day, month, year = parse_date()
    bytes_file = pickle.dumps(model)
    minio_client.put_object(
        bucket_name=bucket_name,
        object_name=f"bitcoin_artifacts/{day}-{month}-{year}/model.pkl",
        data=BytesIO(bytes_file),
        length=len(bytes_file)
    )
    logging.info(
        "Model Artifacts saved to :  bitcoin_artifacts/%s-%s-%s/...", day, month, year)

if __name__ == "__main__":
    main()