# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:36:34 2019

@author: Sumit
"""

# Load libraries
import pandas as pd
from sklearn import svm 
from sklearn import metrics 
    
columns = ['height', 'age', 'weight', 'gender']
train = pd.read_csv("PB1_train.csv", header=None, names=columns) #Reading the training dataset

attributes = ['height', 'age', 'weight']
X_train = train[attributes] # Attributes of the data
y_train = train.gender # Class

test = pd.read_csv("PB1_test.csv", header=None, names=columns) #Reading the test dataset

X_test = test[attributes] # Attributes of the data
y_test = test.gender # Class

print("LINEAR KERNEL FUNCTION-")

SV1 = svm.SVC(kernel="linear",gamma="auto") # Create the svm classifier for linear kernel function
SV1.fit(X_train,y_train) # Training the svm classifier

y_class1 = SV1.predict(X_test) #Predicting the class for test dataset
print("\tClass")
print("Predicted\tActual")
for x in range(20):
    print(y_class1[x],"\t\t",y_test[x]) # Printing the predicted class values for the test data
    
print("Accuracy:",metrics.accuracy_score(y_test, y_class1)*100,"%") # Accuracy of the model

print("\nPOLYNOMIAL FUNCTION WITH DEGREE 5 -")

SV2 = svm.SVC(kernel="poly", degree=5, gamma= "scale") # Create the svm classifier for polynomial kernel function. gamma= "scale" is used to to avoid FutureWarning Messages in scikit-learn
SV2.fit(X_train,y_train) # Training the svm classifier

y_class2 = SV2.predict(X_test) #Predicting the class for test dataset
print("\tClass")
print("Predicted\tActual")
for x in range(20):
    print(y_class2[x],"\t\t",y_test[x]) # Printing the predicted class values for the test data
    
print("Accuracy:",metrics.accuracy_score(y_test, y_class2)*100,"%") # Accuracy of the model

print("\nRADIAL BASIS FUNCTION -")

SV3 = svm.SVC(kernel="rbf",gamma="auto") # Create the svm classifier for rbf kernel function
SV3.fit(X_train,y_train) # Training the svm classifier

y_class3 = SV3.predict(X_test) #Predicting the class for test dataset
print("\tClass")
print("Predicted\tActual")
for x in range(20):
    print(y_class3[x],"\t\t",y_test[x]) # Printing the predicted class values for the test data
    
print("Accuracy:",metrics.accuracy_score(y_test, y_class3)*100,"%") # Accuracy of the model