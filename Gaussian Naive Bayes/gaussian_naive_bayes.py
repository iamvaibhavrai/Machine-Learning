from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

iris = load_iris()
data = iris.data
target = iris.target

dtrain, dtest, ttrain, ttest = train_test_split(data,target,test_size=0.4,random_state=1)

gnb = GaussianNB()
gnb.fit(dtrain,ttrain)

ypred = gnb.predict(xtest)

accuracy = metrics.accuracy_score(ypred,ytest)

print(accuracy*100)
