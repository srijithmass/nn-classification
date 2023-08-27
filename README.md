# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Import the necessary packages & modules

### STEP 2:

Load and read the dataset

### STEP 3:

Perform pre processing and clean the dataset

### STEP 4:

Encode categorical value into numerical values using ordinal/label/one hot encoding

### STEP 5:

Visualize the data using different plots in seaborn

### STEP 6:

Normalize the values and split the values for x and y

### STEP 7:

Build the deep learning model with appropriate layers and depth

### STEP 8:

Predict the values.

## PROGRAM
Programed by: SRIJITH R 212221240054
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("customers.csv")
df.head()
df.isnull().sum()
df.dtypes

df = df.drop('ID',axis=1)
df = df.drop('Var_1',axis=1)
df
df = df.dropna(axis=0)
df

print(df['Gender'].unique())
print(df['Ever_Married'].unique())
print(df['Graduated'].unique())
print(df['Profession'].unique())
print(df['Spending_Score'].unique())
print(df['Segmentation'].unique())

categories_list=[['Male', 'Female'],['No', 'Yes'],
                 ['No', 'Yes'],['Healthcare', 'Engineer',
                 'Lawyer','Artist', 'Doctor','Homemaker',
                 'Entertainment', 'Marketing', 'Executive'],
                 ['Low', 'Average', 'High']]

enc = OrdinalEncoder(categories=categories_list)

df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = enc.fit_transform(df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])
df

scale = MinMaxScaler()
scale.fit(df[["Age"]]) # Fetching Age column alone
df[["Age"]] = scale.transform(df[["Age"]])

df['Segmentation'].unique()

X = df[['Gender','Ever_Married','Age','Graduated', 'Profession','Work_Experience','Spending_Score','Family_Size']].values

y = df[['Segmentation']].values
y

ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)

model = Sequential([Dense(50,input_shape = [8]),
                 Dense(40,activation="relu"),
                 Dense(30,activation="relu"),
                 Dense(20,activation="relu"),
                 Dense(4,activation="softmax")])

model.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

model.fit( x = x_train, y = y_train,
        epochs=500, batch_size=256,
        validation_data=(x_test,y_test),
        )
x_pred = np.argmax(model.predict(x_test), axis=1)
x_pred.shape

y_truevalue = np.argmax(y_test,axis=1)
y_truevalue

```

## Dataset Information

![](o1.png)

![](o2.png)

## OUTPUT

![](o3.png)

## RESULT

A neural network classification model is developed for the given dataset.
