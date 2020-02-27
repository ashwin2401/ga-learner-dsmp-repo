# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
#print(df.head())
#print(df.info())
df['INCOME'] = df['INCOME'].str.replace("$",'',regex=True)
df['HOME_VAL'] = df['HOME_VAL'].str.replace("$",'',regex=True)
df['BLUEBOOK'] = df['BLUEBOOK'].str.replace("$",'',regex=True)
df['OLDCLAIM'] = df['OLDCLAIM'].str.replace("$",'',regex=True)
df['CLM_AMT'] = df['CLM_AMT'].str.replace("$",'',regex=True)
df['INCOME'] = df['INCOME'].replace(",",'',regex=True)
df['HOME_VAL'] = df['HOME_VAL'].replace(",",'',regex=True)
df['BLUEBOOK'] = df['BLUEBOOK'].replace(",",'',regex=True)
df['OLDCLAIM'] = df['OLDCLAIM'].replace(",",'',regex=True)
df['CLM_AMT'] = df['CLM_AMT'].replace(",",'',regex=True)

X = df.drop(['CLAIM_FLAG'],1)
y = df['CLAIM_FLAG']
count = y.value_counts()
print(count)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)
# Code ends here


# --------------
# Code starts here
columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
for col in columns:
    X_train[col] = X_train[col].astype(float)
#print(X_train.info())
for col in columns:
    X_test[col] = X_test[col].astype(float)
#print(X_test.info())
print(X_train.isnull().mean()*100)
print(X_test.isnull().mean()*100)
# Code ends here


# --------------
# Code starts here
#y_train = y_train.set_index[X_train.index]
#y_test = y_test.set_index[X_test.index]
columns = ['AGE','CAR_AGE','INCOME','HOME_VAL']
for col in columns:
    X_train[col].fillna(X_train[col].mean(),inplace=True)
for col in columns:
    X_test[col].fillna(X_test[col].mean(),inplace=True)
X_train.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train = X_train.index
y_test = X_test.index
# Code ends here



# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
for col in columns:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=9)
X_train,y_train = smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
from sklearn.linear_model import LogisticRegression
# Code Starts here
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)
# Code ends here


