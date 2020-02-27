### Project Overview

 As you are working in the insurance company. Company wants to know the reason why claim was not made. Doing so would allow insurance company to improve there policy for giving loan to the customer. In this project you are dealing with various feature such as age, occupation etc. based on that let's get back to the final conculsion.

About the Dataset:
The dataset has details of 10302 Insurance claim with the following 25 features.

Feature	Description
ID	Claim ID
KIDSDRIV	Number of kids person having
AGE	Age of the customer
HOMEKIDS	Number of kids in the home
YOJ	Year of joining of the customer (employee/unemployee)
INCOME	Anual income of the customer
PARENT1	parent is alive or not
HOME_VAL	Home value of the customer
MSTATUS	Marital status
GENDER	Male/Female
EDUCATION	Degree holds by the customer
OCCUPATION	Job title
TRAVTIME	Traveling time
CAR_USE	purpose of the car (private/commercial)
BLUEBOOK	Legal citation system in the United States
CAR_TYPE	Type of car(SUV/Pick up)
RED_CAR	Colour of the car
OLDCLAIM	Old calim of the car
CLM_FREQ	Number of times claims taken
REVOKED	Claim revoked
MVR_PTS	Claim points
CLM_AMT	Claim amount
CAR_AGE	Age of the car
CLAIM_FLAG	Target variable (YES/NO)
Why solve this project?
This is imbalanced dataset . Here 0 - Claim was not made, 1 - Claim made. After completing this project, you will have the better understanding of how to build deal with imbalanced dataset. In this project, you will apply the following concepts.

Train-test split
Standard scaler
Logistic Regression
SMOTE
feature scaling


### Learnings from the project

 I learn how to deal with imbalanced data w.r.t target variable, from end to end data pre-processing , feature engineering and feature scaling.


### Approach taken to solve the problem

 It was a challenging classification problem first i need to deal with missing values , converting the categorical values to numerical values with the help of labelencoder, after that i normalised the data with standard scaler then i applied the models.


