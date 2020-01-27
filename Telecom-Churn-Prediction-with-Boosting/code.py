# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
X = df.drop(['customerID','Churn'],1)
y = df.Churn
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here

#Replacing spaces with 'NaN' in train dataset
X_train['TotalCharges'].replace(' ',np.NaN, inplace=True)

#Replacing spaces with 'NaN' in test dataset
X_test['TotalCharges'].replace(' ',np.NaN, inplace=True)

#Converting the type of column from X_train to float
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)

#Converting the type of column from X_test to float
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

#Filling missing values
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(),inplace=True)
X_test['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace=True)

#Check value counts
print(X_train.isnull().sum())

cat_cols = X_train.select_dtypes(include='O').columns.tolist()

for i in cat_cols:
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])

for i in cat_cols:
    le = LabelEncoder()
    X_test[i] = le.fit_transform(X_test[i])

y_train = y_train.replace({'No':0,'Yes':1})
y_test = y_test.replace({'No':0,'Yes':1})





# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#. Code starts here
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test) 
ada_score = accuracy_score(y_test,y_pred)
ada_cm = confusion_matrix(y_test,y_pred)
ada_cr = classification_report(y_test,y_pred)
print(ada_score)
print(ada_cm)
print(ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test,y_pred)
print(xgb_score)
xgb_cm = confusion_matrix(y_test,y_pred)
print(xgb_cm)
xgb_cr = classification_report(y_test,y_pred)
print(xgb_cr)

print('-----------*20')
clf_model = GridSearchCV(xgb_model,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred =clf_model.predict(X_test)
clf_score = accuracy_score(y_test,y_pred)
print(clf_score)
clf_cm = confusion_matrix(y_test,y_pred)
print(clf_cm) 
clf_cr = classification_report(y_test,y_pred)
print(clf_cr)


