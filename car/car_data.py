import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv("/Users/jilljoshi/Desktop/Data predictions/adidas/dataset_1.data", names=headers)
# print(df.head(15))
df.head(5)
# replace ? with Nan
df.replace("?", np.nan ,inplace= True)
avg_nol_val = df["normalized-losses"].astype(np.float32).mean()
df["normalized-losses"].replace(np.nan, avg_nol_val, inplace=True)

avg_bore=df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore,inplace = True)

avg_stroke=df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

X= df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four", inplace= True)

df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

missing_data = df.isnull()
for column in headers:
    print(column)
    print (missing_data[column].value_counts())
    print("")  
df.to_csv("Car_dataset_no_null")     
    
#data spliting 
 # Load the data from the CSV file
data = pd.read_csv('/Users/jilljoshi/Desktop/Data predictions/adidas/Car_dataset_no_null')

# Assuming your CSV file has columns 'Length', 'Width', 'Height', and 'Price'
X = data[['length', 'width', 'height']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

