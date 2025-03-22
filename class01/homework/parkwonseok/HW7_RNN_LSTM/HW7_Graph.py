import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# Fetching the data
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')

# Selecting and scaling the features
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)

# Isolating the 'Close' column for y
dfy = dfx[['Close']]

# Removing the 'Close' column from dfx for x
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

# Convert to list for further operations if necessary
x = dfx.values.tolist()
y = dfy.values.tolist()  # This should now only contain the 'Close' data

print("Shape of x:", np.array(x).shape)  # Expected shape: (number of records, 4)
print("Shape of y:", np.array(y).shape)  # Expected shape should be (287, 1) if there are 287 trading days in the date range

print(x, y)

window_size = 10
data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i:i+window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)


# Convert lists to numpy arrays for better handling
data_x = np.array(data_x)
data_y = np.array(data_y)

# Calculate the number of data points for each set
total_samples = len(data_x)
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.2)
test_size = total_samples - train_size - val_size

# Split the data into training, validation, and test sets
train_x = data_x[:train_size]
train_y = data_y[:train_size]

val_x = data_x[train_size:train_size+val_size]
val_y = data_y[train_size:train_size+val_size]

test_x = data_x[train_size+val_size:]
test_y = data_y[train_size+val_size:]

print("Training Data:", train_x.shape, train_y.shape)
print("Validation Data:", val_x.shape, val_y.shape)
print("Test Data:", test_x.shape, test_y.shape)

# Number of features is the number of input variables. For example, Open, High, Low, Volume.
n_features = 4  # Adjust this based on your actual features
n_outputs = 1   # This is because we're predicting a single value (e.g., next day's close price)

# Define the model
modelRNN = Sequential([
    SimpleRNN(20, activation='tanh', input_shape=(window_size, n_features), return_sequences=True),
    Dropout(0.1),
    SimpleRNN(20, activation='tanh', return_sequences=False),
    Dropout(0.1),
    Dense(n_outputs)
])

## Define the model
modelLSTM = Sequential([
    LSTM(20, activation='relu', input_shape=(window_size, n_features), return_sequences=True),
    Dropout(0.1),
    LSTM(20, activation='relu', return_sequences=False),
    Dropout(0.1),
    Dense(n_outputs)
])

# Define the GRU model
modelGRU = Sequential([
    GRU(20, activation='relu', input_shape=(window_size, n_features), return_sequences=True),
    Dropout(0.1),
    GRU(20, activation='relu', return_sequences=False),
    Dropout(0.1),
    Dense(n_outputs)
])



# Compile the model
modelRNN.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelLSTM.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelGRU.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
modelRNN.summary()
modelLSTM.summary()
modelGRU.summary()

historyRNN = modelRNN.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)

historyLSTM = modelLSTM.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)

historyGRU = modelGRU.fit(train_x, train_y,
                          validation_data=(val_x, val_y),
                          epochs=70, batch_size=30)


# Extracting the history of training and validation loss, and MAE
train_loss_RNN = historyRNN.history['loss']
val_loss_RNN = historyRNN.history['val_loss']
train_mae_RNN = historyRNN.history['mae']
val_mae_RNN = historyRNN.history['val_mae']

# Extracting the history for LSTM
train_loss_LSTM = historyLSTM.history['loss']
val_loss_LSTM = historyLSTM.history['val_loss']
train_mae_LSTM = historyLSTM.history['mae']
val_mae_LSTM = historyLSTM.history['val_mae']

# Extracting the history for GRU
train_loss_GRU = historyGRU.history['loss']
val_loss_GRU = historyGRU.history['val_loss']
train_mae_GRU = historyGRU.history['mae']
val_mae_GRU = historyGRU.history['val_mae']


epochs = range(1, 71)  # Assuming 70 epochs as specified

# Plotting the training and validation loss
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(epochs, train_loss_RNN, 'b-', label='Training Loss RNN')
plt.plot(epochs, val_loss_RNN, 'b--', label='Validation Loss RNN')
plt.plot(epochs, train_loss_LSTM, 'r-', label='Training Loss LSTM')
plt.plot(epochs, val_loss_LSTM, 'r--', label='Validation Loss LSTM')
plt.plot(epochs, train_loss_GRU, 'g-', label='Training Loss GRU')
plt.plot(epochs, val_loss_GRU, 'g--', label='Validation Loss GRU')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the training and validation MAE
plt.subplot(2, 1, 2)
plt.plot(epochs, train_mae_RNN, 'b-', label='Training MAE RNN')
plt.plot(epochs, val_mae_RNN, 'b--', label='Validation MAE RNN')
plt.plot(epochs, train_mae_LSTM, 'r-', label='Training MAE LSTM')
plt.plot(epochs, val_mae_LSTM, 'r--', label='Validation MAE LSTM')
plt.plot(epochs, train_mae_GRU, 'g-', label='Training MAE GRU')
plt.plot(epochs, val_mae_GRU, 'g--', label='Validation MAE GRU')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions
pred_RNN = modelRNN.predict(test_x)
pred_LSTM = modelLSTM.predict(test_x)
pred_GRU = modelGRU.predict(test_x)

# Flatten test_y for plotting
true_values = test_y.flatten()

# Flatten predictions for plotting
pred_RNN = pred_RNN.flatten()
pred_LSTM = pred_LSTM.flatten()
pred_GRU = pred_GRU.flatten()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(true_values, label='True Values', color='black', linestyle='--')
plt.plot(pred_RNN, label='RNN Predictions', color='blue')
plt.plot(pred_LSTM, label='LSTM Predictions', color='red')
plt.plot(pred_GRU, label='GRU Predictions', color='green')
plt.title('Comparison of RNN, LSTM, and GRU Predictions with True Values')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Stock Price')
plt.legend()
plt.show()
