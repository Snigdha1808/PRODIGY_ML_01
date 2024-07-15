"Dataset from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data "

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = '/content/train.csv'
data = pd.read_csv(file_path)

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'
data_subset = data[features + [target]]

data_subset['TotalBath'] = data_subset['FullBath'] + 0.5 * data_subset['HalfBath']

X = data_subset[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y = data_subset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_test_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)

coefficients = dict(zip(X.columns, linear_model.coef_))
intercept = linear_model.intercept_

print("Linear Regression Model Coefficients:")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef}")
print("\nModel Intercept:")
print(intercept)
print("\nMean Squared Error:")
print(mse)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices')
plt.show()