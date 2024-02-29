import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('Car_dataset_no_null')

# Assuming df is your DataFrame
mean_length = df['height'].mean()

print("Mean Width:", mean_length)
