import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

dirname = os.path.dirname(__file__)
iris_data_file = os.path.join(dirname, "iris.csv")
iris = pd.read_csv(iris_data_file)
# print(iris) # works

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris["Species"]
# print(y)
# print(X)
print("Classes: ", np.unique(iris["Species"]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
# Logistic regression Model
log_model = SGDClassifier(loss='log')
log_model.fit(X_train,y_train)
pred_log_model = log_model.predict(X_test)
# print(pred_log_model)
# print(y_test)
print("-------------------\nLogistic Regr Scores\n-------------------")
accuracy = accuracy_score(pred_log_model, y_test)
print("Accuracy score: ", accuracy)
print("Precision score: ", precision_score(pred_log_model, y_test,average='macro'))
print("Recall score: ", recall_score(pred_log_model, y_test,average='macro'))
print("f1 score: ", f1_score(pred_log_model, y_test,average='macro'))

#Support Vector Machine model
svm_model = SGDClassifier()
svm_model.fit(X_train,y_train)
pred_svm_model = svm_model.predict(X_test)
# print(pred_svm_model)
# print(y_test)
print("-------------------\nSVM Scores\n-------------------")
accuracy = accuracy_score(pred_svm_model, y_test)
print("Accuracy score: ", accuracy)
print("Precision score: ", precision_score(pred_svm_model, y_test,average='macro'))
print("Recall score: ", recall_score(pred_svm_model, y_test,average='macro'))
print("f1 score: ", f1_score(pred_svm_model, y_test,average='macro'))

# Cross Validation
print("-------------------\nCross Validation Score\n-------------------")
from sklearn.model_selection import cross_val_score
log_regr_cross_validation = cross_val_score(log_model,X_train,y_train,cv=5,scoring='accuracy')
print("Logistic regr: \n", log_regr_cross_validation)
svm_cross_validation = cross_val_score(svm_model,X_train,y_train,cv=5,scoring='accuracy')
print("SVM: \n", svm_cross_validation)

# Confusion Matrix: see how many it got confused
print("-------------------\nConfusion Matrix\n-------------------")
from sklearn.metrics import confusion_matrix
# print(y_test)
# Logistic regression one
log_regr_confusion_matrix = confusion_matrix(pred_log_model, y_test)
print("Logistic Regr Confusion Matrix: \n",  log_regr_confusion_matrix)
# Suppport Vector Machines one
svm_confusion_matrix = confusion_matrix(pred_svm_model, y_test)
print("SVM Confusion Matrix: \n",  svm_confusion_matrix)

# EXTRA --------------------------------------------------------------------------------------
print("\n--------------------------------------\nExtra LogisticRegression()\n--------------------------------------")
# Making predictions
from sklearn.preprocessing import StandardScaler

# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(X, y)

diff_log_model = LogisticRegression()
diff_log_model.fit(train_features, train_labels)
# Score the model on the train data
diff_log_model.score(train_features, train_labels)
print(diff_log_model.score(train_features, train_labels))

# Score the model on the test data
print(diff_log_model.score(X_test, y_test))
iris_1 = np.array([4.8, 3.2, 1.5, 0.6])
iris_2 = np.array([6.4, 3.3, 4.5, 1.5])
iris_3 = np.array([7.1, 2.3, 6.1, 2.1])

sample_iris = np.array([iris_1, iris_2, iris_3])

# Make predictions
print("Predictions:\n", diff_log_model.predict(sample_iris))
print("Probability:\n", diff_log_model.predict_proba(sample_iris))