#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import numpy as np
import sys


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(df, categorical):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred


def run():
    taxi_type = sys.argv[1] 
    year = int(sys.argv[2]) 
    month = int(sys.argv[3]) 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{taxi_type}-{year:04d}-{month:02d}.parquet'

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    y_pred = predict(df, categorical)

    df_result = pd.DataFrame()
    df_result['predictions'] = y_pred.tolist()
    df_result['ride_id'] = df['ride_id'].tolist()

    df_result.to_parquet(output_file, 
                         engine='pyarrow', 
                         compression=None,
                         index=False
                         )

    print(np.mean(y_pred))


if __name__ == '__main__':
    run()


