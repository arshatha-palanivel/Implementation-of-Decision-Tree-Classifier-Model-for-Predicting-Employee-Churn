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

![](./01.png)

Data info:

![](./02.png)

Optimization of null values:

![](./03.png)

Assignment of x and y values:

![](./04.png)

Converting string literals to numerical values using label encoder:

![](./05.png)

Accuracy:

![](./06.png)

Prediction:

![](./07.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
