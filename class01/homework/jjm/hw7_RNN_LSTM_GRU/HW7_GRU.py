
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split


def MinMaxScaler(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)

  return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', "High", 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', "High", 'Low', 'Volume']]



x = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10      # 10일치
data_x = []
data_y = []
for i in range(len(y) - window_size):
  _x = x[i : i + window_size]
  _y = y[i + window_size]
  data_x.append(_x)
  data_y.append(_y)

data_X = np.array(data_x)
data_Y = np.array(data_y)

x_train, x_temp, y_train, y_temp = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = False)       # 70 : 30
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = 0.33, shuffle = False)          # 20 : 10


model_input = Input(shape=(10, 4))
x = GRU(units = 20, activation = 'relu', return_sequences = True) (model_input)
x = Dropout(0.1) (x)
x = GRU(units = 20, activation = 'relu') (x)
x = Dropout(0.1) (x)
model_output = Dense(units=1)(x)

model = Model(inputs = model_input, outputs = model_output)     # 모델 생성
model.summary()

model.compile(optimizer = 'adam', loss = 'mse') 
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 70, batch_size = 30)


test_accu = model.evaluate(x_test, y_test)

model.save('./models/Finance_GRU.h5')

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis = 1)
max(y_pred)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.savefig('./src_img/Finance_GRU_01_Loss.png', dpi=300, bbox_inches='tight')
plt.close()



