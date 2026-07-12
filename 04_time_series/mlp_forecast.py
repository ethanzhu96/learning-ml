import torch
import torch.nn as nn
from sine_wave_data import generate_sine_wave, create_windows

epochs = 1000
loss_fn = nn.MSELoss()

series = generate_sine_wave(200, 0.05)
window_size = 20
x, y = create_windows(series, window_size)

model = nn.Sequential(
    nn.Linear(window_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.6f}")

with torch.no_grad():
    predictions = model(x)

print(f"final loss: {loss.item():.6f}")
print("first 10 predictions:", predictions[:10].squeeze().numpy())
print("first 10 actual:", y[:10].squeeze().numpy())
