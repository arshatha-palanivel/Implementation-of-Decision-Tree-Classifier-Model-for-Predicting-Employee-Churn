# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data
2. Define your model
3. Define your cost function
4. Define your learning rate
5. Train your model
6. Evaluate your model
7. Tune hyperparameters
8. Deploy your model

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Arshatha P
RegisterNumber: 212222230012
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
Initial data set:

<img width="507" alt="01" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/1ad2ea4a-d33a-4f60-95f4-058149182594">


Data info:

<img width="157" alt="02" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/e50451d4-086c-4f24-ba34-174277a22a85">


Optimization of null values:

<img width="143" alt="03" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/9695b483-0e2a-41b7-9f4c-475e9e2dc303">


Assignment of x and y values:

<img width="677" alt="04" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/187c49f7-9890-4951-96a8-bac464e5fcab">


Converting string literals to numerical values using label encoder:

<img width="594" alt="05" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/71510749-6be3-4067-889f-786c3de3569e">


Accuracy:

<img width="124" alt="06" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/6ac36645-e346-4272-b2a3-92af31620cd2">


Prediction:

<img width="673" alt="07" src="https://github.com/arshatha-palanivel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118682484/e61727b3-3ebb-47de-9931-00a426c16c82">


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
