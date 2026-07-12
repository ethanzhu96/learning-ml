import numpy as np
import torch

def generate_sine_wave(num_points=200, noise=0.0):
    x = np.linspace(0, 8 * np.pi, num_points)
    y = np.sin(x)

    if noise > 0:
        y = y + noise * np.random.randn(num_points)
    
    return y

def create_windows(series, window_size):
    x = []
    y = []

    for i in range(len(series) - window_size):
        x.append(series[i:i + window_size])
        y.append(series[i + window_size])

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return x, y