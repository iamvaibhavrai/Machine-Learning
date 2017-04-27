from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
# Loading Data and Label
iris = load_iris()

data = iris.data
label = iris.target

# Splitting Data
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size=0.2,random_state=1)

# Training Data
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(data_train,label_train)

# Prediction
label_pred = knc.predict(data_test)

# Checking Accuracy
accuracy = metrics.accuracy_score(label_test,label_pred)
print("Accuracy: ",accuracy)

# Prediction from sample data
sample = [[5.4,3.9,1.7,0.4],[6.5,3.0,5.8,2.2]]
pred = knc.predict(sample)
pred_speceies = [iris.target_names[p] for p in pred]
print("Prediction")
print(pred_speceies)

# Saving model
joblib.dump(knc,"iris_neighbors.pkl")
