import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('Netflix_watching.csv')
df = df.drop(df.columns[[4,5]], axis=1)
df = df.dropna(subset="Hours Viewed")
df["Hours Viewed"] = df["Hours Viewed"].str.replace("," , "").astype(int)
df["Release Date"] = pd.to_datetime(df["Release Date"])
df['Release Date Month'] = df['Release Date'].dt.month
df['Release Date Month-Year'] = df['Release Date'].dt.to_period('M')

df['Available Globally?'] = np.where(df['Available Globally?'] == 'No', 0, 1)
df.dropna(subset=['Release Date'], inplace=True)
df = df.dropna(subset=['Release Date'])
df= df.reset_index(drop=True)
X = df.info()
print(X)

# fig_2 = px.histogram(df,x='Release Date Month',y='Hours Viewed',color='Available Globally?',barmode='group')
# fig_2.show()

# fig = px.histogram(df,x='Release Date',y='Hours Viewed',color='Available Globally?',barmode='group')
# fig.show()




# grouped_data = df.groupby(['Release Date', 'Available Globally?'])['Hours Viewed'].sum().unstack()

# # Plotting
# grouped_data.plot(kind='bar', stacked=True, color=['blue','red'])

# # Setting labels and title
# plt.xlabel('Release Date')
# plt.ylabel('Hours Viewed')
# plt.title('Hours Viewed by Release Date and Availability Globally')

# # Show the plot
# plt.show()

# test