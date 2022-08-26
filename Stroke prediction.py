#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:34:44 2022
@author: suhana

We will learn how to predict stroke using 5 machine learning algorithms
@author: suhana
"""
#STROKE PREDICTION - Training and Testing
import pandas as pd
import matplotlib.pyplot as plt

#import data
data = pd.read_csv('/Users/suhana/Desktop/Stroke Prediction/healthcare-dataset-stroke-data.csv')

data.info()

#see what is null in this data 
data.isnull().sum()

#fill the null values
data['bmi'].value_counts()

data['bmi'].describe()

#average in the missing bmi dataset 
data['bmi'].fillna(data['bmi'].mean(),inplace=True) #inplace permanent 

data['bmi'].describe()

data.isnull().sum()

#removing the id from the dataset as we don't need it
data.drop('id',axis=1,inplace=True)
#axis - 0 - rows and axis 1 is column so we are dropping the id which is column 

#Outlier removal
#remove the outlier for the data to be balanced 

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=800, facecolor='w', edgecolor='k')

data.plot(kind='box')
plt.show()

data['avg_glucose_level'].describe
data.head() 

#2nd video
#Label Encoding - changes words into integer or float
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

gender = enc.fit_transform(data['gender']) #male is going to be 0 and female is going to be 1

smoking_status = enc.fit_transform(data['smoking_status'])

#To see the list of the smoking status of people if it has changed or not
list(smoking_status)

work_type = enc.fit_transform(data['work_type'])

Residence_type = enc.fit_transform(data['Residence_type'])

ever_married = enc.fit_transform(data['ever_married'])


"""
"""

#Replacing the updated work type, ever married, residence type, 
#smoking status and gender with the information of interger or float
data['work_type'] = work_type
data['ever_married'] = ever_married
data['Residence_type'] = Residence_type
data['smoking_status'] = smoking_status
data['gender'] = gender

data 

data.info()

#SPLITTING/PARTIONING THE DATA FOR TRAIN AND TEST 
#Here X means "Features" and Y means target varaible so that is "STROKE"
#We will train and test X in the ratio of 80 to 20 and train and test Y 
#as well
X = data.drop('stroke',axis = 1) 
X.head()

Y = data['stroke']
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 10) 
#ratio - 80 to 20
X_train

Y_train

X_test

Y_test

#NORMALIZE THE DATA
"""
"""
data.describe()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#Training and testing of the data has to be seperate
X_train_std = std.fit_transform(X_train) #Fit transform - understand the data and then transform
X_test_std = std.transform(X_test)

X_train_std
#The Y datas are only 0 and 1 so we do not have to transform Y 

#Let's save the scaler object
"""
"""
import pickle
import os

scaler_path = os.path.join('/Users/suhana/Desktop/Stroke Prediction/',
                         'models/scaler.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std, scaler_file)        #We just created a file named scaler inside the module

X_train_std

X_test_std

"""
"""
#TRAINING So the 5 machine learning algorithms that we are going to use are - 
#1. Decision Tree
#2. Logistic Regression 
#3. K Nearest Neighbors
#4. Random Forest 
#5. SVM 

#Decision Tree - Here the decision tree is going to have different branches for example 
#people who smoke and don't smoke, if they have any heart yes or no 

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train_std,Y_train)

dt.feature_importances_      #feature importance is going help us determine which 
#attribute is important 
#to predict stroke  - here after running this line we can see that bmi, average 
#glucose level and age is the most important attribute

X_train.columns

Y_pred = dt.predict(X_test_std)

Y_pred

from sklearn.metrics import accuracy_score

ac_dt = accuracy_score(Y_test,Y_pred)

ac_dt    #accuracy of decision tree

#Saving the decision tree
import joblib
model_path=os.path.join('/Users/suhana/Desktop/Stroke Prediction/','models/dt.sav')
joblib.dump(dt, model_path)                 #We have dt.sav on our model folder

#Logistic Regression      - works on binary

from sklearn.linear_model import LogisticRegression      #used in classification
lr = LogisticRegression()

lr.fit(X_train_std,Y_train)

Y_pred_lr = lr.predict(X_test_std)

Y_pred_lr

ac_lr = accuracy_score(Y_test,Y_pred_lr)

ac_lr             
"""
"""
#KNN (K Nearest Neighbor)
"""
"""
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn.fit(X_train_std,Y_train)

Y_pred=knn.predict(X_test_std)
ac_knn = accuracy_score(Y_test,Y_pred)

ac_knn
"""
"""

#Random Forest - bunch of decision tree 
"""
"""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train_std,Y_train)
Y_pred = rf.predict(X_test_std)

ac_rf = accuracy_score(Y_test,Y_pred)

ac_rf          #0.9393346379647749

ac_knn      #0.9344422700587084

ac_dt        #0.9041095890410958

ac_lr         #0.9383561643835616 #highest here

"""
"""
#SVM - used for classification     SVM will help us draw a best fit line that classifies between
# two different things 
"""
"""
from sklearn.svm import SVC
sv = SVC()

sv.fit(X_train_std,Y_train)

Y_pred=sv.predict(X_test_std)

ac_sv = accuracy_score(Y_test,Y_pred)

ac_sv      #0.9393346379647749


"""
"""
#Creation of plot to see which one is better in a plot 
plt.bar(['Decision Tree','Logistic','KNN','Random Forest','SVM'],
        [ac_dt,ac_lr,ac_knn,ac_rf,ac_sv])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()

#WEB DEPLOYMENT



















