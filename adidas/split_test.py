import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Assuming your CSV file is in the same directory as your Python script
file_path = 'adidas.csv'

# Read the CSV file
df = pd.read_csv(file_path)
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True, errors='coerce')
df['total_sales'] = df['total_sales'].str.replace(',', '').astype(float)  # Convert to float
df['operating_profit'] = df['operating_profit'].str.replace(',', '').astype(float)  # Convert to float
df['date'] = df['invoice_date'].dt.day
df['year'] = df['invoice_date'].dt.year
df['month'] = df['invoice_date'].dt.month
df['date'] = df['date'].astype(int)
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)
df['sale_date'] = df['date'].astype(str) + df['month'].astype(str) + df['year'].astype(str)

sale = df.groupby(['year', 'month', 'date', 'total_sales'])['operating_profit'].sum().reset_index()
sale1 = df.groupby(['year', 'month', 'date', 'operating_profit'])['total_sales'].sum().reset_index()

X = sale[['year', 'month', 'date', 'total_sales']]
Y = sale['operating_profit']
X1 = sale1[['year', 'month', 'date', 'operating_profit']]
Y1 = sale1['total_sales']

# X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)
X_train, x_test, Y_train, y_test = train_test_split(X1, Y1, test_size=0.2, train_size=0.8, random_state=42)

# model = LinearRegression()
# model.fit(X_train, Y_train)
# y_pred = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print(accuracy)
# validate = pd.DataFrame({'actual': y_test, 'pre': y_pred})
# print(validate.head())
# Continue with the rest of your code

model = LinearRegression()
poly = PolynomialFeatures(degree=3, include_bias=True)
X_train = poly.fit_transform(X_train)
x_test = poly.fit_transform(x_test)
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)
# y_pred = model.predict(poly.fit([2022,1,12,700000]))
# print(x_test)
accuracy = model.score(x_test, y_test)
print(accuracy)
validate = pd.DataFrame({'actual': y_test, 'pre': y_pred})

print(validate.head())
# plot grapj
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='red', label='Actual vs. Predicted')
sns.lineplot(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], color='black', linestyle='solid',
             label='Regression Line')
plt.xlabel('Actual Operating Profit')
plt.ylabel('Predicted Operating Profit')
plt.title('Actual vs. Predicted Operating Profit')
plt.legend()
plt.show()