# from sklearn.datasets import load_iris

# iris = load_iris()

# print iris.feature_names

# print iris.target_names

# print iris.data[0]

# print iris.target[0]


import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

test_idx = [0,50,100]


#training

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx,axis=0)

#testing

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]



clf = tree.DecisionTreeClassifier()

clf = clf.fit(train_data,train_target)


print test_target

print clf.predict(test_data)