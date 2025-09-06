

import pandas as pd

import matplotlib
from sklearn import linear_model

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
from sklearn.metrics import r2_score

import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
print(df.head())
print(df.info())
df.drop(columns=["id","Unnamed: 0"], inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)


print("Number of Unique bedrooms: ",df['bedrooms'].isnull().sum())
print("Number of Unique bathrooms: ",df['bathrooms'].isnull().sum())
FloorValues = df['floors'].value_counts().to_frame()
print(FloorValues)
sns.boxplot(x="waterfront", y="price", data=df)
plt.title('House Prices by Waterfront View')
plt.xlabel('Waterfront (0 = No, 1 = Yes)')
plt.ylabel('Price (USD)')
plt.show()

sns.regplot(x="sqft_above", y="price", data=df,line_kws={'color':'red'})
plt.title('House Prices by Sqmt Above View')
plt.show()

lm = linear_model.LinearRegression()

lm.fit(df[['sqft_living']],df['price'])
print("Coefficient : ",lm.coef_)
print("Intercept: ",lm.intercept_)
r2_multiple = lm.score(df[['sqft_living']],df['price'])
print("R2 Score: ",r2_multiple)
lm.fit(df[["floors",
"waterfront",
"lat",
"bedrooms",
"sqft_basement",
"view",
"bathrooms",
"sqft_living15",
"sqft_above",
"grade",
"sqft_living"]],df['price'])
print("Coefficient : ",lm.coef_)
print("Intercept: ",lm.intercept_)
r2_simple = lm.score(df[["floors",
"waterfront",
"lat",
"bedrooms",
"sqft_basement",
"view",
"bathrooms",
"sqft_living15",
"sqft_above",
"grade",
"sqft_living"]],df['price'])
print("R2 Score: ",r2_simple)

features = [
    "floors", "waterfront", "lat", "bedrooms", "sqft_basement",
    "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"
]

target = "price"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())

])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("After Pipeline")
print("R2 Score: ",r2)
ridge = Ridge(alpha=0.1)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("After Ridge")
print("R2 Score: ",r2)

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.transform(X_test)
ridge = Ridge(alpha=0.1)
ridge.fit(x_train_poly, y_train)
y_pred = ridge.predict(x_test_poly)
r2 = r2_score(y_test, y_pred)
print("After Ridge with polynomial")
print("R2 Score: ",r2)