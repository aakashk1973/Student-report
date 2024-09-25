import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

df.head()

consolidated_score = (df['math_score'] + df['reading_score'] + df['writing_score'])/3
df['consolidated_score'] = consolidated_score

df.head()

df.columns = ["gender","race/ethnicity","parental level of education","lunch","test preparation course","math_score","reading_score","writing_score"]

df.columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

X = df[['gender',	'race/ethnicity', 'parental level of education',	'lunch',	'test preparation course']] # Use a list of column names
y = df['consolidated_score']

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor() 
knn.fit(X_train, y_train)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=21)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

y_pred = knn.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared:', r2)
