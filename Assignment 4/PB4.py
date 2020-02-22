# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:34:41 2019

@author: Sumit
"""



# Load libraries
import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
    
columns = ['height', 'age', 'weight', 'gender']
train = pd.read_csv("PB4_train.csv", header=None, names=columns) #Reading the training dataset

attributes = ['height', 'age', 'weight']
X_train = train[attributes] # Attributes of the data
y_train = train.gender # Class

test = pd.read_csv("PB4_test.csv", header=None, names=columns) #Reading the test dataset

X_test = test[attributes] # Attributes of the data
y_test = test.gender # Class

dt = DecisionTreeClassifier(criterion="gini") # Create the decision tree
dt = dt.fit(X_train,y_train) # Training the decision Tree

y_class = dt.predict(X_test) #Predicting the class for test dataset
print("\tClass")
print("Predicted\tActual")
for x in range(30):
    print(y_class[x],"\t\t",y_test[x]) # Printing the predicted class values for the test data
    
print("Accuracy:",metrics.accuracy_score(y_test, y_class)*100,"%") # Accuracy of the model

task4 = StringIO()
export_graphviz(dt, out_file=task4,
                filled=True, rounded=True,
                special_characters=True, feature_names = attributes, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(task4.getvalue())  
graph.write_png('task4.png')
Image(graph.create_png())