# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:34:41 2019

@author: Sumit
"""


# Load libraries
import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

    
col_names = ['height', 'age', 'weight', 'gender']
# load training dataset
pima = pd.read_csv("PB4_train.csv", header=None, names=col_names)
pima.head()

#split dataset in features and target variable
feature_cols = ['height', 'age', 'weight']
X_train = pima[feature_cols] # Features
y_train = pima.gender # Target variable

# load test dataset
#col_names = ['height', 'age', 'weight', 'gender']
pima_test = pd.read_csv("PB4_test.csv", header=None, names=col_names)
pima_test.head()

#split dataset in features and target variable
#feature_cols = ['height', 'age', 'weight']
X_test = pima_test[feature_cols] # Features
y_test = pima_test.gender # Target variable

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print(y_pred)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('gender.png')
Image(graph.create_png())