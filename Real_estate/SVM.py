import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'your_data.csv' with your actual file)
data = pd.read_csv('house_data.csv')

# Assume 'price' is the target variable
X = data[['bedrooms', 'bathrooms', 'floors', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'waterfront', 'view',
          'condition', 'grade', 'sqft_basement', 'yr_built', 'yr_renovated']]

y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# # Standardize features (optional but can be beneficial for linear models)
# scaler = StandardScaler()
# X_train_poly = scaler.fit_transform(X_train_poly)
# X_test_poly = scaler.transform(X_test_poly)

# Create and train the Support Vector Machine (SVM) model
svm_model = SVR(kernel='linear')  # You can change the kernel as needed (linear, poly, rbf, etc.)
svm_model.fit(X_train_poly, y_train)

# Make predictions on the test set
predictions = svm_model.predict(X_test_poly)

# Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)
accuracy = 100 * svm_model.score(X_test_poly, y_test)

# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')
print(f'Accuracy: {accuracy:.2f}%')
