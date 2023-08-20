import pandas as pd
import numpy as np

df=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf4.csv',index_col=0)

StandardizeCSDdf4NewspaperName1=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\StandardizeCSDdf4NewspaperName1.csv')

merged_df = pd.merge(df1, df2, on='A', how='left')

# Group the dataset by newspaper names and count the unique years for each newspaper
newspaper_years = df.groupby('Newspaper')['Year'].nunique()

# Filter the dataset to include only newspapers with only 2013 data
newspapers_2013_only = newspaper_years[newspaper_years == 1]
newspapers_2013_only.to_csv(r"E:\IOnewspaper\openaipdf\CSD2016\newspapers_2013_only.csv")





df.sort_values(by=['Newspaper','EDITION', 'Year'], inplace=True)  # sort by newspaper and year

df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf4sorted.csv')

sorted_df=df.sort_values(['Newspaper','Market','Owner'])
sorted_df=sorted_df[['Newspaper','Market','Owner']]
unique_sorted_CSDdf4 = sorted_df.drop_duplicates()
unique_sorted_CSDdf4.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\unique_sorted_CSDdf4.csv')


df['first_year'] = df.groupby(['Newspaper','EDITION'])['Year'].transform('min')  # get the first year for each newspaper
df['equal_to_previous'] = df.groupby(['Newspaper','EDITION'])['FreeCirculation'].shift() == df['FreeCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'FreeCirculation'] = np.nan
df['equal_to_previous'] = df.groupby(['Newspaper','EDITION'])['PaidCirculation'].shift() == df['PaidCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'PaidCirculation'] = np.nan
df['equal_to_previous'] = df.groupby(['Newspaper','EDITION'])['TotalCirculation'].shift() == df['TotalCirculation']  # check if current circulation equals to the previous one
df.loc[(df['Year'] != df['first_year']) & df['equal_to_previous'], 'TotalCirculation'] = np.nan

df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf5.csv')





import matplotlib.pyplot as plt


# Create a histogram of the number of newspapers for each market
plt.hist(newspaper_counts[['newspaper_count']], bins='auto', edgecolor='black')

# Set the labels and title
plt.xlabel('Number of Newspapers')
plt.ylabel('Count')
plt.title('Histogram of Number of Newspapers for Each Market')

# Show the plot
plt.show()
