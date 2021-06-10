import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def load(train_test_split=0.3):
    # t: times as integer values
    # ts: timestamps

    # Read in NYC taxi data
    df = pd.read_csv('nyc_taxi.csv', dayfirst=True, parse_dates=True, )
    ts = pd.to_datetime(df["timestamp"])
    x = np.array(df["value"])

    ## Plotting
    # plt.figure()
    # plt.plot(ts,x)
    # plt.show()

    # Normalisation
    x = x * 1/np.max(x)
    t = np.arange(len(x))

    # Seasonal-trend decomposition
    period = 48 * 7 # 48 time points a day (30 min intervals) * 7 days per week
    result = seasonal_decompose(x, model="additive", period = period, two_sided=False) # One sided for streaming application
    res = result.resid
    data = x[res==res]
    x = res[res==res]
    t = t[res==res]
    ts = np.array(ts)[res==res]

    # Train/test split
    s = int(len(t) * train_test_split)
    x_train = x[:s]; t_train = t[:s]; ts_train = ts[:s]
    x_test = x[s:]; t_test = t[s:]; ts_test = ts[s:]
    data_train = data[:s]; data_test = data[s:]

    # Read in anomaly data
    labels =  eval(open("nyc_anomalies.txt").read())
    anomaly_times = labels["nyc_taxi"]
    anomaly_ts = list(pd.to_datetime(anomaly_times))
    anomaly_index = [list(ts_test).index(anomaly) for anomaly in anomaly_ts]
    corrected_anoamly_ts = list(pd.to_datetime(labels["nyc_taxi_corrected"]))
    corrected_anomaly_index = [list(ts_test).index(anomaly) for anomaly in corrected_anoamly_ts]

    # Generate true class labels
    true = np.zeros(len(ts_test))
    for i in anomaly_index:
        true[i] = 1

    # Generate true class labels
    true_corrected = np.zeros(len(ts_test))
    for i in corrected_anomaly_index:
        true_corrected[i] = 1

    return x_train, t_train, ts_train, x_test, t_test, ts_test, anomaly_ts, true, true_corrected, data_train, data_test
