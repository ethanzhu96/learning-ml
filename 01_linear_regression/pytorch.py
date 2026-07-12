import torch

#data
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([5, 8, 11, 14, 17], dtype=torch.float32)

#model parameters, only tracks thru tensor, grad = true
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

#training settings 
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad 
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss}, w: {w:.4f}, b: {b:.4f}")

print(f"final w: {w:.4f}")
print(f"final b: {b:.4f}")
