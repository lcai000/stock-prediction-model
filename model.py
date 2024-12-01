#imports
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

#fetch historical stock data
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2010-01-01', end='2024-11-30')
#display data
print(data.head())

#data preprocessing
data.fillna(method='ffill', inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

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


# Make them the same length by taking the minimum length
min_length = min(len(X_test), len(y_test))
X_test = X_test[:min_length]
y_test = y_test[:min_length]
X_test = X_test[:len(y_test)]  # Ensure X_test has the same number of samples as y_test

#test
print(len(X_test), len(y_test))  # They should match


#define neural network architecture

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

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

#setup training process

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 4
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor).squeeze()

    # Compute loss
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_test_pred = model(X_test_tensor).squeeze()

# Calculate the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error

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


#save the model
torch.save(model.state_dict(), 'stock_predictor.pth')
