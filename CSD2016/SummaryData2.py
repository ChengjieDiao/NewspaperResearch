import pandas as pd
import numpy as np

df=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf4.csv',index_col=0)




df.sort_values(by=['Newspaper', 'Year'], inplace=True)  # sort by newspaper and year
df['first_year'] = df.groupby('Newspaper')['Year'].transform('min')  # get the first year for each newspaper
df['equal_to_previous'] = df.groupby('Newspaper')['FreeCirculation'].shift() == df['FreeCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'FreeCirculation'] = np.nan
df['equal_to_previous'] = df.groupby('Newspaper')['PaidCirculation'].shift() == df['PaidCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'PaidCirculation'] = np.nan
df['equal_to_previous'] = df.groupby('Newspaper')['TotalCirculation'].shift() == df['TotalCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'TotalCirculation'] = np.nan

df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf5.csv')




df['FreeCirculation'] = df.groupby('Newspaper')['FreeCirculation'].apply(lambda x: x.mask(x.shift() == x) if len(x) > 1 else x).reset_index(drop=True)
df['PaidCirculation'] = df.groupby('Newspaper')['PaidCirculation'].apply(lambda x: x.mask(x.shift() == x) if len(x) > 1 else x).reset_index(drop=True)
df['TotalCirculation'] = df.groupby('Newspaper')['TotalCirculation'].apply(lambda x: x.mask(x.shift() == x) if len(x) > 1 else x).reset_index(drop=True)



df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf5.csv')

print(df)



import matplotlib.pyplot as plt


# Create a histogram of the number of newspapers for each market
plt.hist(newspaper_counts[['newspaper_count']], bins='auto', edgecolor='black')

# Set the labels and title
plt.xlabel('Number of Newspapers')
plt.ylabel('Count')
plt.title('Histogram of Number of Newspapers for Each Market')

# Show the plot
plt.show()
