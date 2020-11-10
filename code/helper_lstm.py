import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Importing dependencies
import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime as dt
import time
#plt.style.use('ggplot')
import math
import statsmodels.api as sm
from keras.optimizers import Adam
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
from tensorflow.python.keras import backend as K
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
num_of_layers = Integer(low=2, high=20, name='num_of_layers')
num_hidden_nodes = Integer(low=1, high=200, name='num_hidden_nodes')
activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
batch_size = Integer(low=2, high=16, name='batch_size')
dropout = Real(low=0, high=0.5, prior='uniform',
                         name='dropout')
dimensions = [learning_rate,
              num_of_layers,
              num_hidden_nodes,
              activation,
              batch_size,
              dropout
             ]
             

def get_data(country):
    url = 'alldata/'+country+'.csv'
    dateparse = lambda dates: [pd.datetime.strptime(d, '%d-%m-%Y') for d in dates]
    df = pd.read_csv(url,parse_dates = True,date_parser=dateparse,index_col=1)
    #df['date'] = pd.to_datetime(df.date, format='%d/%m/%Y')
    master_col='active_case'
    active_case=df['confirmed']-df['deaths']-df['recovered']
    #df2=df[['humidity_mean','humidity_std','dew_mean','dew_std','mean_ozone','std_ozone',
    #      'mean_precip','std_precip','mean_tMax','std_tMax','mean_tMin','std_tMin','mean_uv','std_uv']].copy()
    df2=df[['confirmed','deaths','recovered']].copy()
    df2['active_case']=active_case
    first = df2[master_col]
    df2.drop(labels=[master_col], axis=1,inplace = True)
    df2.insert(0, master_col, first)
    return df2
    
def get_data_feature(country,feature):
    url = 'alldata/'+country+'.csv'
    dateparse = lambda dates: [pd.datetime.strptime(d, '%d-%m-%Y') for d in dates]
    df = pd.read_csv(url,parse_dates = True,date_parser=dateparse,index_col=1)
    #df['date'] = pd.to_datetime(df.date, format='%d/%m/%Y')
    master_col='active_case'
    active_case=df['confirmed']-df['deaths']-df['recovered']
    df2=df[['humidity_mean','humidity_std','dew_mean','dew_std','mean_ozone','std_ozone',
          'mean_precip','std_precip','mean_tMax','std_tMax','mean_tMin','std_tMin','mean_uv','std_uv']].copy()
    #df2['active_case']=active_case
    #features=select_feature(df2)
    features=[feature,feature]
    df4=df2[features].copy()
    #first = df4[master_col]
    #df4.drop(labels=[master_col], axis=1,inplace = True)
    #df4.insert(0, master_col, first)
    return df4
def create_model(X_train,learning_rate, num_of_layers,num_hidden_nodes, activation, dropout):
    print(learning_rate, num_of_layers,num_hidden_nodes, activation, dropout)
    # Adding Layers to the model
    model = Sequential()
    model.add(LSTM(X_train.shape[2],input_shape = (X_train.shape[1],X_train.shape[2]),return_sequences = True,
                  activation = activation))
    #print('Ok')
    for i in range(num_of_layers-1):
        #print(i)
        #print(num_hidden_nodes)
        model.add(LSTM(int(num_hidden_nodes),activation = activation,return_sequences = True))
    model.add(LSTM(int(num_hidden_nodes),activation = activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = 'mean_squared_error')
    model.compile(optimizer = optimizers.Adam(lr = learning_rate), metrics=['mean_squared_error'], loss = 'mean_squared_error')
    return model
    
def get_train_test(series):
    # Train Val Test Split
    train_start = dt.date(2020,1,22)
    train_end = dt.date(2020,6,15)
    train_data = series.loc[train_start:train_end]

    val_start = dt.date(2020,6,16)
    val_end = dt.date(2020,7,10)
    val_data = series.loc[val_start:val_end]

    test_start = dt.date(2020,7,11)
    test_end = dt.date(2020,8,3)
    test_data = series.loc[test_start:test_end]

#print(train_data.shape,val_data.shape,test_data.shape)
    # Normalisation
    sc = MinMaxScaler(feature_range=(0, 1))
    train = sc.fit_transform(train_data)
    val = sc.transform(val_data)
    test = sc.transform(test_data)
    return train,val,test,sc
