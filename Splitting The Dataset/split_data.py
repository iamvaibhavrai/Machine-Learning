from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

data = iris.data
label = iris.target

data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.5,random_state=1)

print("Data Train Shape")
print(data_train.shape)
print("Data Test Shape")
print(data_test.shape)
print("Label Train Shape")
print(label_train.shape)
print("Label Test Shape")
print(label_test.shape)
