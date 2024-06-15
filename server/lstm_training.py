import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# Load data 
data = pd.read_csv('Training Data.csv')

# Select features and target
features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'precipitation', 'rain', 'snowfall', 'pressure_msl', 'surface_pressure', 'cloudcover', 'windspeed_10m']
target = 'aqi'
data = data[features + [target]]

# Normalize 
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1])  # Exclude the target variable
        y.append(data[i + time_steps, -1])  # Select only the target variable
    return np.array(X), np.array(y)

# Define time steps
time_steps = 24

# Prepare data for LSTM
X, y = prepare_data(data_normalized, time_steps)

# Split data into train, test, and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=12)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.4, random_state=10)

# Model architecture
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1) 
])

# Compile
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
result = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=100, callbacks=[early_stopping], verbose=2)

# Evaluate
prediction = model.predict(X_test)

# Save the model
model.save("lstm.keras")

# Calculate evaluation metrics
rmse = sqrt(mean_squared_error(Y_test, prediction))
print("RMSE Score is", rmse)

mse = mean_squared_error(Y_test, prediction)
print("MSE Score is", mse)

mean_abs_error = mean_absolute_error(Y_test, prediction)
print("Mean absolute error is", mean_abs_error)

# Calculate and print R-squared value
r2 = r2_score(Y_test, prediction)
print("R-squared is", r2)

# Plot graph of actual vs predicted AQI values
plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Actual AQI')
plt.plot(prediction, label='Predicted AQI', alpha=0.7)
plt.title('Actual vs Predicted AQI')
plt.xlabel('Samples')
plt.ylabel('AQI')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, prediction, alpha=0.7)
plt.title('Scatter plot of Actual vs Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.show()

# Print model architecture
model.summary()
