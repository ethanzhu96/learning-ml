import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype= float)
y = np.array([0, 0, 1, 1, 1], dtype= float)

w = 0.0
b = 0.0

epochs = 1000
learning_rate = 0.05
n = len(x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    z = w * x + b
    y_pred = sigmoid(z)

    loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
    error = y_pred - y

    dw = (1/n) * np.sum(error * x)
    db = (1/n) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"epoch: {epoch}")
        print(f"w: {w:.4f}")
        print(f"b: {b:.4f}")
        print(f"loss: {loss:.4f}")

probabilities = sigmoid(w * x + b)
predictions = (probabilities >= 0.5).astype(int)

print(f"final w: {w:.4f}")
print(f"final b: {b:.4f}")
print(f"final loss: {loss:.4f}")
print("probabilities:", probabilities)
print("predictions:", predictions)


