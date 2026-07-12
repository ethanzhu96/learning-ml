import numpy as np

# data. real relationship should be y = 3x + 2
x = np.array([1, 2, 3, 4, 5], dtype= float)
y = np.array([5, 8, 11, 14, 17], dtype= float)

# starting nums for formula y = wx + b
w = 0.0
b = 0.0

#training settings
learning_rate = 0.01 #amount of update each step. calculated like w = w - learning_rate * dw
epochs = 1000 # number of iterations
n = len(x) #number of elements for computing stuff like MSE

# iterating thru epocs lmao
for epoch in range(epochs):
    y_pred = w * x + b

    error = y_pred - y
    loss = np.mean(error ** 2)

    #compute gradients
    dw = (2 / n) * np.sum((y_pred - y) * x)
    db = (2 / n) * np.sum(y_pred - y)

    # update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss}, w: {w:.4f}, b: {b:.4f}")

print(f"final w: {w:.4f}")
print(f"final b: {b:.4f}")
