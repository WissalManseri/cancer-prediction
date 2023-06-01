import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from log_reg import Logistic_Regression
from kmeans import KMeans

# Loading Inputs (X) and Outputs (Y) data from wdbc.data
X = np.loadtxt("wdbc.data", delimiter=',', usecols=range(2,32))
Y = np.loadtxt("wdbc.data", dtype=str ,delimiter=',', usecols=(1))

# Convert values of Outputs into 1 (M) and 0 (B)
Y = [1 if i == 'M' else 0 for i in Y]


# Split Data into Training and Testing sets
test_size = 0.25
test_rows = round(len(Y) * test_size)
train_rows = len(Y) - test_rows
X_train = X[0:train_rows, :].copy()
X_test = X[train_rows:len(Y) + 1, :].copy()
Y_train = np.array(Y[0:train_rows].copy())
Y_test =  np.array(Y[train_rows:len(Y) + 1].copy())

# plt.figure()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:])
# plt.show()

def confusion_matrix(Y_pred, Y):
    tp = fp = tn = fn = 0
    for i in range (len(Y)):
        if Y[i] == 1 and Y_pred[i] == 1:
            tp += 1
        elif Y[i] == 0 and Y_pred[i] == 0:
            tn += 1
        elif Y[i] == 1 and Y_pred[i] == 0:
            fn += 1
        elif Y[i] == 0 and Y_pred[i] == 1:
            fp += 1
    return tp, fp, tn, fn

print('KNN Model:')
# initializing model
knn_model = KNN(k=2)
# train model
knn_model.fit(X_train, Y_train)
# test model
Y_pred = knn_model.predict(X_test)
# accuracy
tp, fp, tn, fn = confusion_matrix(Y_pred, Y_test)
print(f"tp: {tp}    fp: {fp}    tn: {tn}    fn: {fn} ")
print(f"Accuracy of KNN Model using Confusion matrix: {(tp + tn) / (tp + fp + tn + fn)}")
print(f"Sensitivity : {(tp / (tp+fn))}")
print(f"Specificity : {(tn / (tn+fp))}")

plt.figure(1)
plt.subplot(211)
plt.subplot(211).set_title('Real')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:])
plt.subplot(212)
plt.subplot(212).set_title('Prediction')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred[:])
plt.show()

print('-----------------------')
print('Logistic Regression Model:')
# initializing weights
weights_num = X.shape[1] + 1
weights = [0] * weights_num
# initializing model
lr_model = Logistic_Regression()
# train model
lr_model.fit(X_train, Y_train, weights, 0.01, 1000)
# test model
Y_pred = lr_model.predict(X_test)
# accuracy
tp, fp, tn, fn = confusion_matrix(Y_pred, Y_test)
print(f"tp: {tp}    fp: {fp}    tn: {tn}    fn: {fn} ")
print(f"Accuracy of Logistic Regression Model using Confusion matrix: {(tp + tn) / (tp + fp + tn + fn)}")
print(f"Sensitivity : {(tp / (tp+fn))}")
print(f"Specificity : {(tn / (tn+fp))}")

plt.figure(2)
plt.subplot(211)
plt.subplot(211).set_title('Real')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:])
plt.subplot(212)
plt.subplot(212).set_title('Prediction')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred[:])
plt.show()

print('-----------------------')
print('K-Means Model:')
# initialize and train model
kmeans = KMeans(K=2, max_iters=150)
# test model
Y_pred = kmeans.predict(X)
# accuracy
tp, fp, tn, fn = confusion_matrix(Y_pred, Y_test)
print(f"tp: {tp}    fp: {fp}    tn: {tn}    fn: {fn} ")
print(f"Accuracy of K-Means Model using Confusion matrix: {(tp + tn) / (tp + fp + tn + fn)}")

print(f"Sensitivity : {(tp / (tp+fn))}")
print(f"Specificity : {(tn / (tn+fp))}")

plt.figure(3)
plt.subplot(211)
plt.subplot(211).set_title('Real')
plt.scatter(X[:, 0], X[:, 1], c=Y[:])
plt.subplot(212)
plt.subplot(212).set_title('Prediction')
plt.scatter(X[:, 0], X[:, 1], c=Y_pred[:])
plt.show()



# accuracy = np.sum(Y_pred == Y_test) / len(Y_test)
# print(f"Accuracy of KNN Model: {accuracy}")