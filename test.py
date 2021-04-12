import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"Training Data.csv")
test_data = pd.read_csv(r"Test Data.csv")

train_data.head()
train_data.shape

test_data.head()
test_data.shape

train_data["profession"]=pd.factorize(train_data.profession)[0]
train_data["city"]=pd.factorize(train_data.city)[0]
train_data["state"]=pd.factorize(train_data.state)[0]
train_data["married"]=pd.factorize(train_data.married)[0]
train_data["house_ownership"]=pd.factorize(train_data.house_ownership)[0]
train_data["car_ownership"]=pd.factorize(train_data.car_ownership)[0]

test_data["profession"]=pd.factorize(test_data.profession)[0]
test_data["city"]=pd.factorize(test_data.city)[0]
test_data["state"]=pd.factorize(test_data.state)[0]
test_data["married"]=pd.factorize(test_data.married)[0]
test_data["house_ownership"]=pd.factorize(test_data.house_ownership)[0]
test_data["car_ownership"]=pd.factorize(test_data.car_ownership)[0]

xtrain=train_data.drop("risk_flag",axis=1)
ytrain=train_data["risk_flag"]

from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(xtrain,ytrain)

y_pred= DTClassifier.predict(test_data)

print(y_pred)