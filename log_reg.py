import numpy as np

class Logistic_Regression:
    def __init__(self):
        self.w = []

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # logistic function
    def hx(self, w, X):
        weights_num = X.shape[1] + 1
        z_arrays = [1] * weights_num
        z_arrays[0] = w[0]
        for i in range(weights_num - 1):
            z_arrays[i + 1] = w[i + 1] * np.array(X[:, i])
        z = np.array(sum(z_arrays[j] for j in range(weights_num)))
        return self.sigmoid(z)

    # cost function - binary cross entropy
    def cost(self, w, X, Y):
        y_pred = self.hx(w, X)
        return -1 * sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

    # gradient descent
    def grad_descent(self, w, X, Y):
        y_pred = self.hx(w, X)
        weights_num = X.shape[1] + 1
        grad = [0] * weights_num
        grad[0] = -1 * sum(Y * (1 - y_pred) - (1 - Y) * y_pred)
        for i in range(weights_num - 1):
            grad[i + 1] = -1 * sum(Y * (1 - y_pred) * X[:, i] - (1 - Y) * y_pred * X[:, i])
        return grad

    def fit(self, X, Y, w, lr, max_iter):
        weights_num = X.shape[1] + 1
        iter = 0
        while iter < max_iter:
            w_prev = w.copy()
            grad = self.grad_descent(w_prev, X, Y)
            for index in range(weights_num):
                w[index] = w[index] - lr * grad[index]
            iter += 1
        self.w = w.copy()

    def predict(self, X):
        predictions = self.hx(self.w, X)
        return np.around(predictions).astype(int)
