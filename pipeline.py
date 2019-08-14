from sklearn import datasets

iris = datasets.load_iris()



X = iris.data
Y = iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)


#  decision tree method
# from sklearn import tree

# clf = tree.DecisionTreeClassifier()

# clf.fit(X_train, Y_train)


# predictions = clf.predict(X_test)

# print predictions


# Accuracy calculation

# from sklearn.metrics import accuracy_score

# print accuracy_score(Y_test, predictions)



# Kneighbor method

from sklearn.neighbors import KNeighborsClassifier

clf_kn = KNeighborsClassifier()

clf_kn.fit(X_train, Y_train)

predict_kn = clf_kn.predict(X_test)

print predict_kn

from sklearn.metrics import accuracy_score

print accuracy_score(Y_test, predict_kn)


