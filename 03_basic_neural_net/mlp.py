import torch 
import torch.nn as nn

x = torch.tensor([
    [0,0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0],
], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

epochs = 1000
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.4f}")

probabilities = model(x)
predictions = (probabilities >= 0.5).int()

print(f"final loss: {loss.item():.4f}")
print("probabilities:", probabilities.detach().numpy())
print("predictions:", predictions.detach().numpy())
