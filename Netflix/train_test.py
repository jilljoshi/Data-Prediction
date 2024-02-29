import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv('Netflix_watching.csv')

df = df.drop(df.columns[[4,5]], axis=1)
df = df.dropna(subset="Hours Viewed")
df["Hours Viewed"] = df["Hours Viewed"].str.replace("," , "").astype(int)
df["Release Date"] = pd.to_datetime(df["Release Date"])
df['Month'] = df['Release Date'].dt.month
df['Year'] = df['Release Date'].dt.year
df['Available Globally?'] = np.where(df['Available Globally?'] == 'No', 0, 1)
df.dropna(subset=['Release Date'], inplace=True)
df = df.dropna(subset=['Release Date'])
df= df.reset_index(drop=True)
df = df.drop("Release Date", axis=1)

X = df[['Month', 'Year', 'Available Globally?']]
y = df['Hours Viewed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
numeric_features = ['Month', 'Year']

categorical_features = ['Available Globally?']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Create and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_regressor.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# X = df.head(10)
# print(X)
