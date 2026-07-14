import torch
import torch.nn as nn
from sine_wave_data import generate_sine_wave, create_windows

epochs = 1000
loss_fn = nn.MSELoss()

series = generate_sine_wave(200, 0.05)
window_size = 20
x, y = create_windows(series, window_size)
x = x.unsqueeze(-1)
 # print(x.shape) -> verify the LSTM input shape is correct

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #making lstm layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        #making linear layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_output = lstm_output[:, -1, :]
        prediction = self.linear(last_output)
        return prediction
    
    model = LSTMModel()