import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

csv_data = pd.read_csv('breastCancer.csv')

print('The data frame has', csv_data.shape[0], 'rows and', csv_data.shape[1], 'columns.')
csv_data.info()
print(csv_data.head(5))

csv_data.drop(csv_data.columns[[-1, 0]], axis=1, inplace=True)

diagnosisAll = list(csv_data.shape)[0]
diagnosisCategories = list(csv_data['diagnosis'].value_counts())
print('The data has', diagnosisAll, 'diagnosis,', diagnosisCategories[0], 'malignant and', diagnosisCategories[1], 'benign.')

featuresMean = list(csv_data.columns[1:11])

######### Correlation plot #############

plt.figure(figsize=(10, 10))
sns.heatmap(csv_data[featuresMean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


colorSet = {'M':'red', 'B':'blue'}
colors = csv_data['diagnosis'].map(lambda x: colorSet.get(x))
sm = pd.scatter_matrix(csv_data[featuresMean], c=colors, alpha=0.4, figsize=((10, 10)))
plt.show()


bins = 12
plt.figure(figsize=(10, 10))
for i, feature in enumerate(featuresMean):
    rows = int(len(featuresMean) / 2)
    plt.subplot(rows, 2, i + 1)
    sns.distplot(csv_data[csv_data['diagnosis'] == 'M'][feature], bins=bins, color='red', label='M')
    sns.distplot(csv_data[csv_data['diagnosis'] == 'B'][feature], bins=bins, color='blue', label='B')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

################ Box-Plot #############

plt.figure(figsize=(15, 15))
for i, feature in enumerate(featuresMean):
    rows = int(len(featuresMean) / 2)
    plt.subplot(rows, 2, i + 1)
    sns.boxplot(x='diagnosis', y=feature, data=csv_data, palette="Set1")
plt.tight_layout()
plt.show()


########## Selected features #################

selectedFeatures = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']


setDiagnosis = {'M':1, 'B':0}
csv_data['diagnosis'] = csv_data['diagnosis'].map(setDiagnosis)

X = csv_data.loc[:,featuresMean]
y = csv_data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
accuracy_all_features = []
cv_all_features = []


################## Support Vector Machine #######################

clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_all_features.append(accuracy_score(prediction, y_test))
cv_all_features.append(np.mean(scores))
print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("SVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_all_features.append(accuracy_score(prediction, y_test))
cv_all_features.append(np.mean(scores))
print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("NuSVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_all_features.append(accuracy_score(prediction, y_test))
cv_all_features.append(np.mean(scores))
print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("LinearSVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


#################### Random Forest ####################################

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_all_features.append(accuracy_score(prediction, y_test))
cv_all_features.append(np.mean(scores))
print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Radnom Forest Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


#################### Using Selected Features #########################


X = csv_data.loc[:,selectedFeatures]
y = csv_data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
accuracy_selected_features = []
cv_selected_features = []

print('------------------Results after feature selection-------------')

################## Support Vector Machine #######################

clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_selected_features.append(accuracy_score(prediction, y_test))
cv_selected_features.append(np.mean(scores))
print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("SVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_selected_features.append(accuracy_score(prediction, y_test))
cv_selected_features.append(np.mean(scores))
print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("NuSVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_selected_features.append(accuracy_score(prediction, y_test))
cv_selected_features.append(np.mean(scores))
print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("LinearSVC Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


#################### Random Forest ####################################


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=10)
accuracy_selected_features.append(accuracy_score(prediction, y_test))
cv_selected_features.append(np.mean(scores))
print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Random Forest Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))


accuracy_diff = list(np.array(accuracy_selected_features) - np.array(accuracy_all_features))
cv_diff = list(np.array(cv_selected_features) - np.array(cv_all_features))
d = {'accuracy_all_features':accuracy_all_features, 'accuracy_selected_features':accuracy_selected_features, 'accuracy_diff':accuracy_diff,
     'cv_all_features':cv_all_features, 'cv_selected_features':cv_selected_features, 'cv_diff':cv_diff}
indexList = ['SVC', 'NuSVC', 'LinearSVC', 'RandomForest']
df = pd.DataFrame(d, index=indexList)
print(df)

#################### accuracy comparison graph ######################

ax = df[['accuracy_all_features', 'accuracy_selected_features']].plot(kind='bar', title = "Accuracy comparison", figsize=(5, 5), legend=True, fontsize=10)
ax.set_xlabel("Classification Algorithm", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
plt.show()


ax = df[['cv_all_features', 'cv_selected_features']].plot(kind='bar', title = "Cross-validation  comparison", figsize=(5, 5), legend=True, fontsize=10)
ax.set_xlabel("Classification Algorithm", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
plt.show()

