import pandas as pd

data = pd.read_csv("weather.csv")

shape = data.shape
feature = data.columns

feature_matrix = data[data.columns[:-1]]
response_vectore = data[data.columns[-1]]

print("Shape",shape)
print("Feature",feature)
print("Feature Matrix")
print(feature_matrix.head())
print("Response Vector")
print(response_vectore.head())
