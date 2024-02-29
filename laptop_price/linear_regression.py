import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns


df = pd.read_csv('Laptop_price.csv')

# correlation matrix
df_corr = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']
sns.heatmap(df[df_corr].corr(), annot = True, cmap = 'magma')
plt.title('Correlation Matrix')
plt.show()

# Extracting features (X) and target variable (y)
X = df[['Processor_Speed','RAM_Size', 'Storage_Capacity', 'Weight']]
y = df['Price']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()
# poly = PolynomialFeatures(degree=2)
# X_train = poly.fit_transform(X_train)
# X_test = poly.fit_transform(X_test)

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy =100*(model.score(X_test, y_test)) 

print("Accuracy", accuracy)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing the predictions
plt.scatter(X_test['Processor_Speed'], y_test, color='black', label='Actual Prices')
plt.scatter(X_test['Processor_Speed'], y_pred, color='blue', linewidth=3, label='Predicted Prices')
plt.xlabel('Processor Speed')
plt.ylabel('Price')
plt.legend()
plt.show()


# Assuming you have your new laptop features in a DataFrame named 'new_laptop'
new_laptop_features = pd.DataFrame({
    'Processor_Speed': [3.2416270596922305],
    'RAM_Size': [4],
    'Storage_Capacity': [256],
    'Weight': [2.029060648222903]
})

# Making predictions for the new laptop
new_laptop_price = model.predict(new_laptop_features)

print('Predicted Price for the New Laptop:', new_laptop_price[0])
