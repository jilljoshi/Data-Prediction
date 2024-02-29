import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read CSV file into a DataFrame
file_path = '/Users/jilljoshi/Desktop/Data predictions/adidas/adidas.csv'  # Update with your actual CSV file path
data = pd.read_csv(file_path)

data['operating_profit'] = data['operating_profit'].replace({',': ''}, regex=True).astype(float)

# Convert 'total_sales' to float and remove commas
data['total_sales'] = data['total_sales'].replace({',': ''}, regex=True).astype(float)

# Assuming your CSV has columns 'X' and 'Y' for independent and dependent variables
X = data['operating_profit'].values.reshape(-1, 1)
y = data['total_sales'].values.data

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions using the model
predictions = model.predict(X)

# Plot the original data points
plt.scatter(X, y, label='Actual data')

# Plot the regression line
plt.plot(X, predictions, color='red', label='Linear Regression')

# Add labels and a legend
plt.xlabel('operating_profit (X)')
plt.ylabel('total_sales (Y)')
plt.title('Linear Regression Plot')
plt.legend()

# Show the plot
plt.show()


