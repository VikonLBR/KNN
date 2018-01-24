import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


'''
an example of iris dataset


iris = datasets.load_iris()
# print(type(iris.data))
dataset = iris.data
classes = iris.target


X_train, X_test, Y_train, Y_test = train_test_split(dataset, classes, test_size=0.2) #test size is the percentage of getting validation
# make the data in the dataset to be random
# print(Y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.predict(X_test) == Y_test)
'''

'''
example of LinearRegression with housing price in boston

boston_data = datasets.load_boston()
data_X = boston_data.data
data_Y = boston_data.target



model = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.1)

model.fit(X_train, Y_train)

fig = plt.figure()
m = len(Y_test)
fake_X = np.arange(0, m)
plt.scatter(fake_X, model.predict(X_test), c='r')
plt.scatter(fake_X, Y_test, c='b')
plt.show()


'''


X, Y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, bias=3, noise=10)
fig = plt.figure()
plt.scatter(X,Y)
plt.show()













