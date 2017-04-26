# Read Documentation First

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
label = iris.target

feature = iris.feature_names
target = iris.target_names

print("Feature",feature)
print("Target",target)
print("Type of data",type(data))
print("Data",data)
