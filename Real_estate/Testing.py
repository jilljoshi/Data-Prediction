# Create new data for testing (replace values with your actual test data)
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

new_data = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'floors': [1],
    'sqft_living': [1500],
    'sqft_lot': [5000],
    'sqft_above': [1200],
    'sqft_basement': [300],
    'waterfront': [0],
    'view': [1],
    'condition': [3],
    'grade': [7],
    'sqft_basement': [0],
    'yr_built': [1990],
    'yr_renovated': [0]
})

# Standardize features if necessary (uncomment if you used StandardScaler during training)
# new_data_scaled = scaler.transform(new_data)

# Transform features if necessary (uncomment if you used PolynomialFeatures during training)
new_data_poly = poly.transform(new_data)

# Make predictions on the new data
new_predictions = model.predict(new_data_poly)

# Display the predicted prices
print("Predicted Price:", new_predictions[0])
