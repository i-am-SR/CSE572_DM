# -*- coding: utf-8 -*-
"""
Created on Sat Apr  13 18:14:15 2019

@author: Sumit
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB   
from sklearn import metrics 
    
columns = ['height', 'age', 'weight', 'gender']
train = pd.read_csv("PB1_train.csv", header=None, names=columns) #Reading the training dataset

attributes = ['height', 'age', 'weight']
X_train = train[attributes] # Attributes of the data
y_train = train.gender # Class

test = pd.read_csv("PB1_test.csv", header=None, names=columns) #Reading the test dataset

X_test = test[attributes] # Attributes of the data
y_test = test.gender # Class

NV = GaussianNB() # Create the Gaussian Classifier
NV.fit(X_train,y_train) # Training the decision Tree

y_class = NV.predict(X_test) #Predicting the class for test dataset
print("\tClass")
print("Predicted\tActual")
for x in range(20):
    print(y_class[x],"\t\t",y_test[x]) # Printing the predicted class values for the test data
print("Accuracy:",metrics.accuracy_score(y_test, y_class)*100,"%") # Accuracy of the model
