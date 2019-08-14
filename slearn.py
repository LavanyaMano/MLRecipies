from sklearn import tree

features = [[150,1],[170,1],[130,0],[140,0]]

label = [0,0,1,1]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, label)


print clf.predict([[140,0]])