# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np


dataset = pd.read_csv('finaly1.csv')

decision = ['area','num_bed','num_bath','school','hospital','marketplace','pharmacy','supermarket','university','price','point']
decision1 = ['area','num_bed','num_bath','school','hospital','marketplace','pharmacy','supermarket','university','point']

a = dataset[decision]
b = a.dropna()
y = b.price
X = b[decision1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)


import joblib

price_predict_model_file = "price_predict_model_file.pkl"
joblib.dump(regressor, price_predict_model_file)



