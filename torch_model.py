#imports
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch historical stock data
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2010-01-01', end='2024-11-30')
# Display data
print(data.head())
print(data.shape)

# Data preprocessing
data.fillna(method='ffill', inplace=True)

train_size = int(len(data) * 0.8)
data_train, data_test = data[:train_size], data[train_size:]

# Normalize data
scaler = StandardScaler()
data_scaled_train = scaler.fit_transform(data_train[['Open', 'High', 'Low', 'Close', 'Volume']])
scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Create a DataFrame from scaled data
scaled_df = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Feature engineering
scaled_df['Prev_Close'] = scaled_df['Close'].shift(1)  # Create lag feature
scaled_df['Next_Close'] = scaled_df['Close'].shift(-1)  # Target: next day's closing price
scaled_df.dropna(inplace=True)  # Drop rows with missing values

# Prepare input and output
X = scaled_df[['Prev_Close', 'Open', 'High', 'Low', 'Volume']].values
y = scaled_df['Next_Close'].values  # Ensure alignment after dropna()

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define neural network architecture
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 64  # Hidden layer size
output_size = 1  # Predicting the next day's price

# Initialize the model
model = StockPredictor(input_size, hidden_size, output_size)

# Setup training process
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor).squeeze()

    # Compute loss
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # every 10 epochs print
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Test the model
# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_test_pred = model(X_test_tensor).squeeze()

# Calculate the Mean Squared Error (MSE)
y_test_pred = y_test_pred[:len(y_test)]  # Slice predictions
mse = mean_squared_error(y_test, y_test_pred.numpy())
print(f"Test Mean Squared Error: {mse:.4f}")

# Predict next day's price (example)
model.eval()
with torch.no_grad():
    next_day_price = model(X_test_tensor[-1:]).item()

# Convert the predicted price back to the original scale
predicted_price = scaler.inverse_transform([[0, 0, 0, next_day_price, 0]])[0][3]  # Inverse scale for the 'Close' column
print(f"Predicted next day's price: {predicted_price:.2f}")

# Fixing the inverse scaling and plot actual vs predicted stock prices
# We only inverse scale the 'Close' column, since that is the column we're predicting

# Inverse scale predictions and actual values (only 'Close')
y_test_actual = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test), 4)), y_test)))[:, 4]
y_test_pred_scaled = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test_pred), 4)), y_test_pred.numpy())))[:, 4]

# Plotting the actual vs predicted prices
plt.figure(figsize=(14, 5))
plt.plot(data_test.index, y_test_actual, label='Actual Stock Price')
plt.plot(data_test.index, y_test_pred_scaled, label='Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

