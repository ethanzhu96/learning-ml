import torch

y = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32)
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

epochs = 2000
learning_rate = 0.01

def sigmoid (z):
    return 1 / (1 + torch.exp(-z))

for epoch in range(epochs):
    z = w * x + b
    y_pred = sigmoid(z)

    loss = -torch.mean(y * torch.log(y_pred) + (1-y) * torch.log(1 - y_pred))
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad 
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 200 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

probabilities = sigmoid(w * x + b)
predictions = (probabilities >= 0.5).int()

print(f"final w: {w.item():.4f}")
print(f"final b: {b.item():.4f}")
print(f"final loss: {loss.item():.4f}")
print("probabilities:", probabilities.detach().numpy())
print("predictions:", predictions.detach().numpy())
