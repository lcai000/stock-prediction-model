import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yfinance as yf

class StockDataset:
    def __init__(self, data, window_size=30, predict_size=1):
        """
        Prepare sequential data for stock price prediction
        
        Parameters:
        - data: numpy array or pandas DataFrame of stock prices
        - window_size: number of previous days to use for prediction
        - predict_size: number of future days to predict
        """
        self.window_size = window_size
        self.predict_size = predict_size
        
        # Convert to numpy if pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Normalize the data
        self.scaler = self._get_scaler(data)
        scaled_data = self.scaler.fit_transform(data)
        
        # Prepare sequential data
        self.X, self.y = self._create_sequences(scaled_data)
    
    def _get_scaler(self, data):
        return StandardScaler()
    
    def _create_sequences(self, data):
        """
        Create sliding window sequences
        
        Returns:
        - X: Input sequences (batch, window_size, features)
        - y: Target values (batch, predict_size)
        """
        X, y = [], []
        
        for i in range(len(data) - self.window_size - self.predict_size + 1):
            # Create input sequence
            input_seq = data[i:i+self.window_size]
            
            # Create target (next day's value)
            target_seq = data[i+self.window_size:i+self.window_size+self.predict_size]
            
            X.append(input_seq)
            y.append(target_seq)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return X, y
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Create a PyTorch DataLoader
        """
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    
    def inverse_transform(self, predictions):
        """
        Convert predictions back to original scale
        """
        return self.scaler.inverse_transform(predictions)

# Example usage
def prepare_stock_data(csv_path):
    # Load stock data
    df = pd.read_csv(csv_path)
    
    # Assume you want to use 'Close' price for prediction
    # You can modify this to use multiple features
    data = df['Close'].values.reshape(-1, 1)
    
    # Create dataset
    stock_dataset = StockDataset(data, window_size=30, predict_size=1)
    
    # Get DataLoader
    train_loader = stock_dataset.get_dataloader(batch_size=32)
    
    return stock_dataset, train_loader

# Neural Network Model
class LSTMStockPredictor(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM expects input: (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class MLPStockPredictor(torch.nn.Module):
    def __init__(self, input_size=30, hidden_size1=64, hidden_size2=32, output_size=1):
        super().__init__()
        self.model = torch.nn.Sequential(
            # First hidden layer
            torch.nn.Linear(input_size, hidden_size1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size1),
            torch.nn.Dropout(0.2),
            
            # Second hidden layer
            torch.nn.Linear(hidden_size1, hidden_size2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size2),
            torch.nn.Dropout(0.2),
            
            # Output layer
            torch.nn.Linear(hidden_size2, output_size)
        )
    
    def forward(self, x):
        # Flatten the input if it's a sequence
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)
        return self.model(x)


# Training function
def train_model(model, train_loader, epochs=50, learning_rate=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze())
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def download_stock_data(ticker, start_date='2005-01-01', end_date=None):
    """
    Download stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock symbol (e.g., 'AAPL' for Apple)
    - start_date: Start date for data collection
    - end_date: End date for data collection (defaults to current date)
    
    Returns:
    - pandas DataFrame with stock data
    """
    import datetime
    
    # Use current date if no end date specified
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if stock_data.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}")
        
        return stock_data
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return None

def train_and_evaluate_model(
    stock_data, 
    ticker, 
    feature_column='Close', 
    window_size=30, 
    test_size=0.2, 
    epochs=100, 
    learning_rate=0.001
):
    """
    Comprehensive model training and evaluation function
    
    Parameters:
    - stock_data: DataFrame with stock prices
    - ticker: Stock symbol
    - feature_column: Which column to use for prediction
    - window_size: Number of previous days to use for prediction
    - test_size: Proportion of data to use for testing
    - epochs: Number of training epochs
    - learning_rate: Optimizer learning rate
    
    Returns:
    - Trained model
    - Performance metrics
    """
    # Prepare data
    data = stock_data[feature_column].values.reshape(-1, 1)
    
    # Create dataset
    stock_dataset = StockDataset(data, window_size=window_size, predict_size=1)
    
    # Split data into train and test
    X, y = stock_dataset.X, stock_dataset.y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    # model = LSTMStockPredictor(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    model = MLPStockPredictor()
    
    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        test_epoch_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.squeeze())
                test_epoch_loss += loss.item()
        
        # Store losses
        train_losses.append(train_epoch_loss / len(train_loader))
        test_losses.append(test_epoch_loss / len(test_loader))
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Test Loss: {test_losses[-1]:.4f}')
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title(f'{ticker} Stock Price Prediction - Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Predictions on test set
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            batch_pred = model(X_batch)
            predictions.extend(batch_pred.numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert back to original scale
    predictions = stock_dataset.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = stock_dataset.scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Visualization of predictions
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.7)
    plt.title(f'{ticker} Stock Price - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    return model, {
        'mse': mse,
        'mae': mae,
        'mape': mape
    }


def predict_next_day(model, stock_data, window_size=30, feature_column='Close'):
    """
    Predict the next day's stock price
    
    Parameters:
    - model: Trained PyTorch model
    - stock_data: DataFrame with historical stock prices
    - window_size: Number of previous days to use for prediction
    - feature_column: Column to use for prediction
    
    Returns:
    - Predicted next day's stock price
    """
    # Prepare the input sequence
    data = stock_data[feature_column].values.reshape(-1, 1)
    
    # Create dataset (reuse the scaling from training)
    stock_dataset = StockDataset(data, window_size=window_size, predict_size=1)
    
    # Get the last window of data
    last_window = data[-window_size:].reshape(1, window_size, 1)
    
    # Convert to PyTorch tensor
    last_window_tensor = torch.FloatTensor(last_window)
    
    # Set model to evaluation mode
    model.eval()
    
    # Predict
    with torch.no_grad():
        predicted_scaled = model(last_window_tensor)
        
        # Inverse transform to get actual price
        predicted_price = stock_dataset.scaler.inverse_transform(
            predicted_scaled.numpy()
        )[0][0]
    
    return predicted_price

# Updated main function to demonstrate prediction
def main(ticker='AAPL'):
    # Download stock data
    stock_data = download_stock_data(ticker)
    
    if stock_data is not None:
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(stock_data, ticker)
        
        # Predict next day's price
        next_day_price = predict_next_day(model, stock_data)
        
        print(f"\nPredicted {ticker} Stock Price for Next Day: ${next_day_price:.2f}")
        
        # Compare with last known price
        last_price = float(stock_data['Close'].iloc[-1])
        print(f"Last Known Price: ${last_price:.2f}")
        
        # Calculate predicted change
        price_change = next_day_price - last_price
        change_percentage = (price_change / last_price) * 100
        
        print(f"Predicted Price Change: ${price_change:.2f}")
        print(f"Predicted Percentage Change: {change_percentage:.2f}%")


if __name__ == "__main__":
    main('AAPL')