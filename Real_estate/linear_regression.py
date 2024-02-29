# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset (replace 'your_data.csv' with your actual file)
data = pd.read_csv('house_data.csv')

# Assume 'price' is the target variable
X = data[ ['bedrooms', 'bathrooms', 'floors', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'waterfront', 'view',
     'condition', 'grade', 'sqft_basement', 'yr_built', 'yr_renovated']]

y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize features (optional but can be beneficial for linear models)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Create and train the Linear Regression model
model = LinearRegression()
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
# print(X_test[0])

model.fit(X_train, y_train)

# Make predictions on the test 
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
accuracy = 100*(model.score(X_test, y_test))
 
print("Accuracy:" , accuracy)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

residuals = y_test - predictions

# Visualize Residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=predictions, y=residuals)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# import pandas as pd

# df = pd.read_csv("house_data.csv")
# header = ["id","date","price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15",'sqft_lot15']
# missing_data = df.isnull()
# for column in header:
#     print(column)
#     print(missing_data[column].value_counts())
#     print("dtype: ", df.dtypes[column])
#     print(" ")




# # X = df.head(5)
# # print(X)
