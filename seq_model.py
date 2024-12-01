import torch
import numpy as np
import pandas as pd

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
        from sklearn.preprocessing import StandardScaler
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