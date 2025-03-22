import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
import tensorflow as tf

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
#"""최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]



x = dfx.values.tolist() # open, high, log, volume, 데이터
y = dfy.values.tolist()

#ex) 1월 1일 ~ 1월 10일까지의 OHLV 데이터로 1월 11일 종가 (Close) 예측
#ex) 1월 2일 ~ 1월 11일까지의 OHLV 데이터로 1월 12일 종가 (Close) 예측
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]
    # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])
test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])
print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

# LSTM

# model_LSTM = Sequential()
# model_LSTM.add(LSTM(units=20, activation='tanh',
# return_sequences=True,
# input_shape=(10, 4)))
# model_LSTM.add(Dropout(0.1))
# model_LSTM.add(LSTM(units=20, activation='tanh'))
# model_LSTM.add(Dropout(0.1))
# model_LSTM.add(Dense(units=1))
# model_LSTM.summary()
# model_LSTM.compile(optimizer='adam',
# loss='mean_squared_error')
# history = model_LSTM.fit(train_x, train_y,validation_data = (val_x, val_y), epochs=70, batch_size=30)


# GRU

# model_GRU = Sequential()
# model_GRU.add(GRU(units=20, activation='tanh',
# return_sequences=True,
# input_shape=(10, 4)))
# model_GRU.add(Dropout(0.1))
# model_GRU.add(GRU(units=20, activation='tanh'))
# model_GRU.add(Dropout(0.1))
# model_GRU.add(Dense(units=1))
# model_GRU.summary()
# model_GRU.compile(optimizer='adam',
# loss='mean_squared_error')
# history = model_GRU.fit(train_x, train_y,
# validation_data = (val_x, val_y),
# epochs=70, batch_size=30)   


# RNN

# model_RNN = Sequential()
# model_RNN.add(SimpleRNN(units=20, activation='tanh',
# return_sequences=True,
# input_shape=(10, 4)))
# model_RNN.add(Dropout(0.1))
# model_RNN.add(SimpleRNN(units=20, activation='tanh'))
# model_RNN.add(Dropout(0.1))
# model_RNN.add(Dense(units=1))
# model_RNN.summary()
# model_RNN.compile(optimizer='adam',
# loss='mean_squared_error')
# history = model_RNN.fit(train_x, train_y,
# validation_data = (val_x, val_y),
# epochs=70, batch_size=30)


model_RNN = model = tf.keras.models.load_model('/home/park/workspace/intel-03/class01/homework/parkwonseok/HW7_RNN_LSTM/models/HW7_RNN.h5')
model_LSTM = model = tf.keras.models.load_model('/home/park/workspace/intel-03/class01/homework/parkwonseok/HW7_RNN_LSTM/models/HW7_LSTM.keras')
model_GRU = model = tf.keras.models.load_model('/home/park/workspace/intel-03/class01/homework/parkwonseok/HW7_RNN_LSTM/models/HW7_GRU.keras')


# Model summary
model_RNN.summary()
model_LSTM.summary()
model_GRU.summary()


RNN_predictions = model_RNN.predict(val_x)
GRU_predictions = model_GRU.predict(val_x)
LSTM_predictions = model_LSTM.predict(val_x)

# 실제 값 (val_y)과 예측 값 비교 차트 그리기
x = np.arange(len(val_y))

plt.figure(figsize=(10, 6))
plt.plot(x, val_y, label='Actual', color='black', marker='o')
plt.plot(x, RNN_predictions, label='RNN Predictions', color='blue', marker='x')
plt.plot(x, GRU_predictions, label='GRU Predictions', color='green', marker='x')
plt.plot(x, LSTM_predictions, label='LSTM Predictions', color='red', marker='x')

plt.title('Model Predictions vs Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()