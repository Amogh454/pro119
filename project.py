import pandas as pd
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

column_names = ["Id", "Class", "Sex", "Age", "SibSp", "Parch", "Survived"]
data=pd.read_csv("titanic.csv", names=column_names).iloc[1:]

features = ["Class", "Sex", "Age", "Parch"]
X=data[features]
Y=data.Survived

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(Xtrain, Ytrain)
Ypredict = clf.predict(Xtest)
accuracy=metrics.accuracy_score(Ytest, Ypredict)*100
print("Accuracy of prediction in percentage: ", accuracy, "%")

data1 = StringIO()
export_graphviz(clf, out_file=data1, feature_names=features, class_names=["0", "1"], filled=True, rounded=True)
print("data for decision Tree: ", data1.getvalue())

graph = pydotplus.graph_from_dot_data(data1.getvalue())
graph.write_png("decisiontree.png")
Image(graph.create_png())