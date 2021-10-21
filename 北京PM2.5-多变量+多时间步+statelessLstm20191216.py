# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 08:28:40 2018

@author: Administrator
"""

#北京PM2.5预测-多变量+多时间步+stateless+lstm

    
#stateful->stateless:
#LSTM:batch_input_shape->input_shape,stateful=True->False
#model.fit(shuffle=False->True,for i in range(N)model.fit(epoch=1),model_reset_state()->model.fit epoch=N)

from matplotlib import pyplot
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM
from math import sqrt
from pandas import read_csv,DataFrame,concat
from sklearn.metrics import mean_squared_error

timeSteps=1
featureDim=8
batchSize=72


# load dataset
filename="C:/Users/Administrator/Desktop/pollution.csv"
dataset = read_csv(filename, header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
    pyplot.show() 

#解决显示省略号问题    
import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)    
#print(dataset.head())
#print(values.shape)

# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)
    
#序列转变成有监督数据
seq_length = timeSteps
dataX = []
dataY = []
for i in range(0, len(values) - seq_length,1):
    seq_in = values[i:i + seq_length,:]
    seq_out = values[i + seq_length,0]
    dataX.append(seq_in)
    dataY.append(seq_out)
dataX=np.array(dataX)
dataY=np.array(dataY)
print(dataX.shape)

#分割数据集
trainX,testX,trainY,testY=train_test_split(dataX,dataY,test_size=0.3,random_state=1)
print(trainX.shape,trainY.shape)

##build model:stack-Lstm
#model = Sequential()
#model.add(LSTM(50,return_sequences=True,input_shape=(trainX.shape[1], trainX.shape[2]),stateful=False))
#model.add(LSTM(50))
#model.add(Dense(50))
#model.add(Dense(1))
#
##compile model
#model.compile(loss='mae',optimizer='adam')
#
## fit network
#history = model.fit(trainX, trainY, epochs=100, batch_size=batchSize, validation_data=(testX, testY), verbose=2, shuffle=False)
#print(history.history['loss'])
##plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
#   
##RMSE评估模型
#predY = model.predict(testX)
#testX = testX.reshape((testX.shape[0], testX.shape[2]))
#
## invert scaling for forecast
#inv_predY = concatenate((predY, testX[:, 1:]), axis=1)
#inv_predY = scaler.inverse_transform(inv_predY)
#inv_predY = inv_predY[:,0]
#
## invert scaling for actual
## invert scaling for actual
#testY = testY.reshape((len(testY), 1))
#inv_testY = concatenate((testY, testX[:, 1:]), axis=1)
#inv_testY = scaler.inverse_transform(inv_testY)
#inv_testY = inv_testY[:,0]
#
## calculate RMSE
#rmse = sqrt(mean_squared_error(inv_testY, inv_predY))
#print('Test RMSE: %.3f' % rmse)
#
#
#
#
