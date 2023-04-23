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
from dotenv import load_dotenv
from minio import Minio

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

def get_btc_data():
    # Get Bitcoin data
    data = yf.download(tickers='BTC-USD', period = '60d', interval = '15m')
    return data

def get_eth_data():
    # Get ETH data
    data = yf.download(tickers='ETH-USD', period = '60d', interval = '15m')
    return data

def main():
    load_dotenv()
    start_time = time.time()
    bucket_name, access_key_id, secret_access_key, minio_api_url = get_minio_credentials()
    minio_client = create_minio_client(minio_api_url, access_key_id, secret_access_key, bucket_name)
    logging.info("Fetching BTC Data ...")
    btc_data = get_btc_data()
    btc_bytes = btc_data.to_csv().encode('utf-8')
    day, month, year = parse_date()
    minio_client.put_object(
        bucket_name,
        f"bitcoin_data/{day}-{month}-{year}/bitcoin.csv",
        data=BytesIO(btc_bytes),
        length=len(btc_bytes),
        content_type="application/csv",
    )
    logging.info(
            "Data saved successfully in : bitcoin_data/%s-%s-%s/...", day, month, year
        )
    logging.info("Fetching ETH Data ...")
    eth_data = get_eth_data()
    eth_bytes = eth_data.to_csv().encode('utf-8')
    minio_client.put_object(
        bucket_name,
        f"etherium_data/{day}-{month}-{year}/etherium.csv",
        data=BytesIO(eth_bytes),
        length=len(eth_bytes),
        content_type="application/csv",
    )
    logging.info(
            "Data saved successfully in : etherium_data/%s-%s-%s/...", day, month, year
        )
if __name__ == "__main__":
    main()