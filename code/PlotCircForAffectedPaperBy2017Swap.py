import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from scipy import stats
import ast
import pickle

import matplotlib
import matplotlib.pyplot as plt
# Assuming that 'owner' is a column in your data


df = pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\Combined.csv', parse_dates=['Year'])

cols_to_drop = df.columns[df.columns.str.startswith('Unnamed')]
df.drop(columns=cols_to_drop,inplace=True)


with open(r'E:\IOnewspaper\openaipdf\CSD2016\affected_market.pkl', 'rb') as f:
    affected_market = pickle.load(f)

#start = df.columns.get_loc('Population and dwelling counts / Population, 2016')
#end = df.columns.get_loc('Immigration / Total Sex / Total / Immigrant status and period of immigration for the population in private households / 25% sample data / Immigrants')

#df.drop(df.columns[start:end+1], axis=1, inplace=True)


#df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\check.csv')
df['CSDs'] = df['CSDs'].apply(ast.literal_eval)

df.sort_values(by=['ID2','Year'],inplace=True)

# Create a temporary DataFrame
temp_df = df.groupby('ID2')['Owner'].transform(lambda x: all(item in ['postmedia network inc.','sun media corporation'] for item in x))
# Merge this DataFrame back to the original DataFrame on 'ID2'
df = pd.concat([df, temp_df.rename('postmedia_temp')], axis=1)
# Set 'postmedia' to 1 where appropriate
df['postmedia'] = 0
df.loc[df['postmedia_temp'], 'postmedia'] = 1
# Drop the temporary column
df = df.drop(columns=['postmedia_temp'])


temp_df = df.groupby('ID2')['Owner'].transform(lambda x: all(item in ['metroland media group ltd.'] for item in x))
# Merge this DataFrame back to the original DataFrame on 'ID2'
df = pd.concat([df, temp_df.rename('metroland_temp')], axis=1)
# Set 'postmedia' to 1 where appropriate
df['metroland'] = 0
df.loc[df['metroland_temp'], 'metroland'] = 1
# Drop the temporary column
df = df.drop(columns=['metroland_temp'])


temp_df = df.groupby('ID2')['Owner'].transform(lambda x: all(item in ['postmedia network inc.','sun media corporation','metroland media group ltd.'] for item in x))
# Merge this DataFrame back to the original DataFrame on 'ID2'
df = pd.concat([df, temp_df.rename('MetrolandOrPostmedia_temp')], axis=1)
# Set 'postmedia' to 1 where appropriate
df['MetrolandOrPostmedia'] = 0
df.loc[df['MetrolandOrPostmedia_temp'], 'MetrolandOrPostmedia'] = 1
# Drop the temporary column
df = df.drop(columns=['MetrolandOrPostmedia_temp'])


print(df[df['Owner']=='metroland media group ltd.']['ID2'].nunique())

print(df[df['postmedia']==1]['ID2'].nunique())
print(df[df['metroland']==1]['ID2'].nunique())
print(df[df['MetrolandOrPostmedia']==1]['ID2'].nunique())
print(df['ID2'].nunique())






df.sort_values(by=['ID2','Year'],inplace=True)
check = df.loc[df['Swap2017']==1,['ID2','Year','Swap2017','MarketAndNewspaper']]
check.drop_duplicates(subset='ID2', keep='last', inplace=True)
check.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\check.csv')
# Assuming your DataFrame is df









import pickle
element_list = pd.read_pickle(r'E:\IOnewspaper\openaipdf\CSD2016\affected_market.pkl')

CDSmapCD = pd.read_pickle(r'E:\IOnewspaper\openaipdf\CSD2016\CDSmapCD.pkl')

# Create a CD list for the corresponding values of element_list
CD_list = [CDSmapCD.get(csd) for csd in element_list]


df['CDs'] = df['CDs'].apply(ast.literal_eval)

df['OwnerPostmediaMetroland'] = df['Owner'].isin(['postmedia network inc.', 'metroland media group ltd.','sun media corporation']).astype(int)

df['Owner2']=df['Owner']

df.loc[df['Owner'].isin(['postmedia network inc.','sun media corporation']),'Owner2']='postmedia network inc.';
df.loc[df['Owner'].isin(['metroland media group ltd.']),'Owner2']='metroland media group ltd.';
df.loc[~df['Owner'].isin(['postmedia network inc.', 'metroland media group ltd.','sun media corporation']),'Owner2']='Others';

df_2017 = df[df['Year'].dt.year== 2017]  # filter for year 2017
df_2018 = df[df['Year'].dt.year== 2018]  # filter for year 2018

# List of elements of interest


def count_unique_newspapers(df_year, elements_of_interest):
    # explode the CSDs column
    df_exploded = df_year.explode('CSDs')

    # filter for elements of interest
    df_filtered = df_exploded[df_exploded['CSDs'].isin(elements_of_interest)]

    # drop duplicates to count unique newspapers only
    df_unique = df_filtered.drop_duplicates(subset=['ID2', 'CSDs', 'Owner2'])

    # group by 'CSDs' and 'Owner2' to count unique newspapers
    result_by_company = df_unique.groupby(['CSDs', 'Owner2']).ID2.nunique().reset_index().rename(
        columns={'ID2': 'Unique Newspaper Count by Company'})

    # pivot the result to make 'Owner2' columns
    pivot_by_company = result_by_company.pivot_table(index='CSDs', columns='Owner2',
                                                     values='Unique Newspaper Count by Company').reset_index().fillna(0)

    return pivot_by_company


# get results for each year
result_2017 = count_unique_newspapers(df_2017, element_list)
result_2018 = count_unique_newspapers(df_2018, element_list)


column_sums2017 = result_2017.sum(axis=0)  # Calculate the sum of each column
# Add the new row with column sums to the dataframe
result_2017.loc[len(result_2017)] = column_sums2017


# Ensure all CSDs in result_2017 are in result_2018
result_2018 = result_2018.set_index('CSDs').reindex(result_2017.CSDs).reset_index().fillna(0)


column_sums2018 = result_2018.sum(axis=0)  # Calculate the sum of each column
result_2018.loc[len(result_2018)] = column_sums2018

result_2017.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\result_2017.csv')
result_2018.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\result_2018.csv')

Pset1=set(result_2017[result_2017['postmedia network inc.']>0]['CSDs'].to_list())
Pset2=set(result_2018[result_2018['postmedia network inc.']>0]['CSDs'].to_list())

Pintersection=Pset1.intersection(Pset2)



Mset1=set(result_2017[result_2017['metroland media group ltd.']>0]['CSDs'].to_list())
Mset2=set(result_2018[result_2018['metroland media group ltd.']>0]['CSDs'].to_list())

Mintersection=Mset1.intersection(Mset2)



with open(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Pintersection.pkl', 'wb') as file:
    pickle.dump(Pintersection, file)

with open(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Mintersection.pkl', 'wb') as file:
    pickle.dump(Mintersection, file)




# Update the function name and variable names
def count_unique_newspapers_cds(df_year, elements_of_interest):
    # explode the CDs column
    df_exploded = df_year.explode('CDs')

    # filter for elements of interest
    df_filtered = df_exploded[df_exploded['CDs'].isin(elements_of_interest)]

    # drop duplicates to count unique newspapers only
    df_unique = df_filtered.drop_duplicates(subset=['ID2', 'CDs', 'Owner2'])

    # group by 'CDs' and 'Owner2' to count unique newspapers
    result_by_company = df_unique.groupby(['CDs', 'Owner2']).ID2.nunique().reset_index().rename(
        columns={'ID2': 'Unique Newspaper Count by Company'})

    # pivot the result to make 'Owner2' columns
    pivot_by_company = result_by_company.pivot_table(index='CDs', columns='Owner2',
                                                     values='Unique Newspaper Count by Company').reset_index().fillna(0)

    return pivot_by_company


# get results for each year using the modified function
result_2017_cds = count_unique_newspapers_cds(df_2017, CD_list)
result_2018_cds = count_unique_newspapers_cds(df_2018, CD_list)

column_sums2017_cds = result_2017_cds.sum(axis=0)  # Calculate the sum of each column
# Add the new row with column sums to the dataframe
result_2017_cds.loc[len(result_2017_cds)] = column_sums2017_cds

# Ensure all CDs in result_2017_cds are in result_2018_cds
result_2018_cds = result_2018_cds.set_index('CDs').reindex(result_2017_cds.CDs).reset_index().fillna(0)


column_sums2018_cds = result_2018_cds.sum(axis=0)  # Calculate the sum of each column
# Add the new row with column sums to the dataframe
result_2018_cds.loc[len(result_2018_cds)] = column_sums2018_cds

# Update the file names in the to_csv function calls
result_2017_cds.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\result_2017_cds.csv', index=False)
result_2018_cds.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\result_2018_cds.csv', index=False)




# Initialize two empty dataframes for results

results_2017 = pd.DataFrame(
    columns=['Element', 'ID2', 'Owner', 'MarketAndNewspaper', 'Year', 'Swap2017Affected', 'Swap2017'])
results_2018 = pd.DataFrame(
    columns=['Element', 'ID2', 'Owner', 'MarketAndNewspaper', 'Year', 'Swap2017Affected', 'Swap2017'])

# Check each element in the element_list
for element in element_list:
    # Find rows in each year where CSDs contain the element
    rows_2017 = df_2017[df_2017['CSDs'].apply(lambda csds: element in csds)]
    rows_2018 = df_2018[df_2018['CSDs'].apply(lambda csds: element in csds)]

    # Add to the respective results DataFrame
    for _, row in rows_2017.iterrows():
        results_2017.loc[len(results_2017)] = [element, row['ID2'], row['Owner'], row['MarketAndNewspaper'],
                                               row['Year'], row['Swap2017Affected'], row['Swap2017']]
    for _, row in rows_2018.iterrows():
        results_2018.loc[len(results_2018)] = [element, row['ID2'], row['Owner'], row['MarketAndNewspaper'],
                                               row['Year'], row['Swap2017Affected'], row['Swap2017']]


results_2017.sort_values(by=['Element','ID2'],inplace=True)
results_2018.sort_values(by=['Element','ID2'],inplace=True)


results_2017.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\check2017.csv')
results_2018.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\check2018.csv')




with open(r'E:\IOnewspaper\openaipdf\CSD2016\affected_market.pkl', 'rb') as f:
    list_of_elements = pickle.load(f)
counts_2017 = []
counts_2018 = []
counts_2017P = []
counts_2018P = []
counts_2017M = []
counts_2018M = []

df['Year'] = pd.to_datetime(df['Year']).dt.year  # Make sure the 'Year' column is in the correct format
df['OwnerPostmediaMetroland'] = df['Owner'].isin(['postmedia network inc.', 'metroland media group ltd.','sun media corporation']).astype(int)




for element in list_of_elements:
    unique_ids_2017 = set()
    unique_ids_2018 = set()
    unique_ids_2017P = set()
    unique_ids_2018P = set()
    unique_ids_2017M = set()
    unique_ids_2018M = set()

    for index, row in df[df['Year'] == 2017].iterrows():
        if element in row['CSDs']:
            unique_ids_2017.add(row['ID2'])
    for index, row in df[df['Year'] == 2018].iterrows():
        if element in row['CSDs']:
            unique_ids_2018.add(row['ID2'])

    for index, row in df[(df['Year'] == 2017) & ((df['Owner']=='postmedia network inc.') | (df['Owner']=='sun media corporation'))].iterrows():
        if element in row['CSDs']:
            unique_ids_2017P.add(row['ID2'])
    for index, row in df[(df['Year'] == 2018) & ((df['Owner']=='postmedia network inc.') | (df['Owner']=='sun media corporation'))].iterrows():
        if element in row['CSDs']:
            unique_ids_2018P.add(row['ID2'])

    for index, row in df[(df['Year'] == 2017) & (df['Owner']=='metroland media group ltd.')].iterrows():
        if element in row['CSDs']:
            unique_ids_2017M.add(row['ID2'])
    for index, row in df[(df['Year'] == 2018) & (df['Owner']=='metroland media group ltd.')].iterrows():
        if element in row['CSDs']:
            unique_ids_2018M.add(row['ID2'])

    counts_2017.append(len(unique_ids_2017))
    counts_2018.append(len(unique_ids_2018))

    counts_2017P.append(len(unique_ids_2017P))
    counts_2018P.append(len(unique_ids_2018P))

    counts_2017M.append(len(unique_ids_2017M))
    counts_2018M.append(len(unique_ids_2018M))


labels = list_of_elements




x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, counts_2017, width, label='2017')
rects2 = ax.bar(x + width/2, counts_2018, width, label='2018')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Elements')
ax.set_ylabel('Counts')
ax.set_title('Counts by element and year')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

# Function to add a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()




x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

# Adjust bar positions
rects1P = ax.bar(x - 3*width/2, counts_2017P, width, label='2017P')
rects2P = ax.bar(x - width/2, counts_2018P, width, label='2018P')
rects1M = ax.bar(x + width/2, counts_2017M, width, label='2017M')
rects2M = ax.bar(x + 3*width/2, counts_2018M, width, label='2018M')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Elements')
ax.set_ylabel('Counts')
ax.set_title('Counts by element and year')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

# Function to add a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1P)
autolabel(rects2P)
autolabel(rects1M)
autolabel(rects2M)

fig.tight_layout()
plt.show()















x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, counts_2017P, width, label='2017')
rects2 = ax.bar(x + width/2, counts_2018P, width, label='2018')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Elements')
ax.set_ylabel('Counts')
ax.set_title('Counts by element and year')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

# Function to add a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()







x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, counts_2017M, width, label='2017')
rects2 = ax.bar(x + width/2, counts_2018M, width, label='2018')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Elements')
ax.set_ylabel('Counts')
ax.set_title('Counts by element and year')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

# Function to add a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()




df['OnlyFree']=0
df.loc[(df['EverFree']==1)&(df['EverPaid']==0),'OnlyFree']=1
df['OnlyPaid']=0
df.loc[(df['EverPaid']==1)&(df['EverFree']==0),'OnlyFree']=1
df['BothPaidFree']=0
df.loc[(df['EverPaid']==1)&(df['EverFree']==1),'BothPaidFree']=1
## Describe the Market
df['Owner']=df['Owner'].str.strip()
# Convert the "Year" column to datetime format
df['Year'] = pd.to_datetime(df['Year'])
df['Time'] = df['Year'].dt.year.astype(int) - 2013
# Create a binary variable for whether the year is after 2017
df['Post2017'] = (df['Year'].dt.year > 2017).astype(int)
# Create a dummy variable for whether owner is 'Postmedia' or 'Metroland'
df['OwnerPostmediaMetroland'] = df['Owner'].isin(['postmedia network inc.', 'metroland media group ltd.','sun media corporation']).astype(int)
print('Total Newspaper:', df['ID'].nunique())
print('Newspaper Owned By OwnerPostmediaMetroland:', df[df['OwnerPostmediaMetroland']==1]['ID'].nunique())
print('Total Newspaper:', df['ID'].nunique())
print('Newspaper Editions Affected by 2017:', df[(df['Swap2017Affected']==1)]['ID'].nunique())
print('Total Newspaper:', df['ID'].nunique())
print('Newspaper Owned By OwnerPostmediaMetroland and Affected by 2017:', df[(df['OwnerPostmediaMetroland']==1)&(df['Swap2017Affected']==1)]['ID'].nunique())


# Assuming df is your DataFrame and it has been defined somewhere above this line of code

## plot those csd cross different years
CrossSubset=df[df['OpCross2017']==1]
# Group the data by year and calculate statistics
grouped_data = CrossSubset.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
# Plot the mean circulation per year for all data
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='All Newspaper Editions In the Subgroup')
# Filter data for Swap2017Affected
Swap2017Affected = CrossSubset[CrossSubset['Swap2017Affected'] == 1]
grouped_data = Swap2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by Swap2017 In the Subgroup')
# Plot the mean circulation per year for Swap2017Affected
OwnedByPM=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==1]
grouped_data = OwnedByPM.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Owned By Postmedia or Metroland In the Subgroup')

OwnedByPM2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==1)&(CrossSubset['Swap2017Affected']==1)]
grouped_data = OwnedByPM2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by 2017 Swap and Owned By Postmedia or Metroland In the Subgroup')


NotOwnedByPM=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==0]
grouped_data = NotOwnedByPM.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Not Owned By Postmedia or Metroland In the Subgroup')


NotOwnedByPM2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==0)&(CrossSubset['Swap2017Affected']==1)]
grouped_data = NotOwnedByPM2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by 2017 Swap and Not Owned By Postmedia or Metroland In the Subgroup')


plt.ylabel('Circulation')
plt.title('Mean Circulation for Newspaper Editions of Each Year')
# Place legend below the figure
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()  # Adjusts the spacing to prevent overlapping
plt.show()



# Group the data by year and calculate statistics
grouped_data = CrossSubset.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='All Newspaper Editions In the Subgroup')
Swap2017Affected=CrossSubset[CrossSubset['Swap2017Affected']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Affected by Swap2017 In the Subgroup')
Swap2017Affected=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Owned By Postmedia or Metroland In the Subgroup')
Swap2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==1)&(CrossSubset['Swap2017Affected']==1)]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Affected by 2017 Swap and Owned By Postmedia or Metroland In the Subgroup')

NotOwnedByPM=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==0]
grouped_data = NotOwnedByPM.groupby('Year')['NatlLineRate']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Not Owned By Postmedia or Metroland In the Subgroup')


NotOwnedByPM2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==0)&(CrossSubset['Swap2017Affected']==1)]
grouped_data = NotOwnedByPM2017Affected.groupby('Year')['NatlLineRate']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by 2017 Swap and Not Owned By Postmedia or Metroland In the Subgroup')

plt.ylabel('National Line Rate')
plt.title('National Line Rate for Newspaper Editions of Each Year')

# Place legend below the figure
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()  # Adjusts the spacing to prevent overlapping
plt.show()





# Group the data by year and calculate statistics
grouped_data = df.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
# Plot the mean circulation per year for all data
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='All Newspaper Editions In the Data')
# Filter data for Swap2017Affected
Swap2017Affected = df[df['Swap2017Affected'] == 1]
grouped_data = Swap2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by Swap2017')
# Plot the mean circulation per year for Swap2017Affected
OwnedByPM=df[df['OwnerPostmediaMetroland']==1]
grouped_data = OwnedByPM.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Owned By Postmedia or Metroland')
OwnedByPM2017Affected=df[(df['OwnerPostmediaMetroland']==1)&(df['Swap2017Affected']==1)]
grouped_data = OwnedByPM2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions Affected by 2017 Swap and Owned By Postmedia or Metroland')
plt.xlabel('Year')
plt.ylabel('Circulation')
plt.title('Mean Circulation for Newspaper Editions of Each Year')
plt.legend()
plt.show()



# Group the data by year and calculate statistics
grouped_data = df.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='All Newspaper Editions In the Data')

Swap2017Affected=df[df['Swap2017Affected']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Affected by Swap2017')

Swap2017Affected=df[df['OwnerPostmediaMetroland']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Owned By Postmedia or Metroland')

Swap2017Affected=df[(df['OwnerPostmediaMetroland']==1)&(df['Swap2017Affected']==1)]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Affected by 2017 Swap and Owned By Postmedia or Metroland')

plt.xlabel('Year')
plt.ylabel('National Line Rate')
plt.title('Mean NatlLineRate for Newspaper Editions of Each Year')
plt.legend()


FreeNewspaper=CrossSubset[CrossSubset['OnlyFree']==1]
grouped_data = FreeNewspaper.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
# Plot the mean circulation per year for all data
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='All Free Newspaper Editions Survive After 2017 In the Data')
# Filter data for Swap2017Affected
Swap2017Affected = FreeNewspaper[FreeNewspaper['Swap2017Affected'] == 1]
grouped_data = Swap2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Free Newspapers Editions Survive After 2017 and Affected by Swap2017')
# Plot the mean circulation per year for Swap2017Affected
OwnedByPM=FreeNewspaper[FreeNewspaper['OwnerPostmediaMetroland']==1]
grouped_data = OwnedByPM.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Free Newspapers Editions Survive After 2017 and Owned By Postmedia or Metroland')
OwnedByPM2017Affected=FreeNewspaper[(FreeNewspaper['OwnerPostmediaMetroland']==1)&(FreeNewspaper['Swap2017Affected']==1)]
grouped_data = OwnedByPM2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Free Newspapers Editions Survive After 2017, Affected by 2017 Swap and Owned By Postmedia or Metroland')
plt.xlabel('Year')
plt.ylabel('Circulation')
plt.title('Mean Circulation for Newspaper Editions of Each Year')
plt.legend()
plt.show()



grouped_data = FreeNewspaper.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='All Newspaper Editions Survive After 2017 In the Data')

Swap2017Affected=FreeNewspaper[FreeNewspaper['Swap2017Affected']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017 and Affected by Swap2017')

Swap2017Affected=FreeNewspaper[FreeNewspaper['OwnerPostmediaMetroland']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017 and Owned By Postmedia or Metroland')

Swap2017Affected=FreeNewspaper[(FreeNewspaper['OwnerPostmediaMetroland']==1)&(FreeNewspaper['Swap2017Affected']==1)]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017, Affected by 2017 Swap and Owned By Postmedia or Metroland')

plt.xlabel('Year')
plt.ylabel('National Line Rate')
plt.title('Mean NatlLineRate for Newspaper Editions of Each Year')
plt.legend()
plt.show()



## plot the newspaper existing post 2017, including those operating before 2017


OpPost2017=df[df['OpPost2017']==1]
grouped_data = OpPost2017.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
# Plot the mean circulation per year for all data
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='All Free Newspaper Editions OpPost2017 In the Data')
# Filter data for Swap2017Affected
Swap2017Affected = OpPost2017[OpPost2017['Swap2017Affected'] == 1]
grouped_data = Swap2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions OpPost2017 and Affected by Swap2017')
# Plot the mean circulation per year for Swap2017Affected
OwnedByPM=OpPost2017[OpPost2017['OwnerPostmediaMetroland']==1]
grouped_data = OwnedByPM.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Free Newspapers Editions OpPost2017 and Owned By Postmedia or Metroland')
OwnedByPM2017Affected=OpPost2017[(OpPost2017['OwnerPostmediaMetroland']==1)&(OpPost2017['Swap2017Affected']==1)]
grouped_data = OwnedByPM2017Affected.groupby('Year')['TotalCirculation']
mean_circulation = grouped_data.mean()
plt.plot(mean_circulation.index, mean_circulation, 'o-', label='Newspapers Editions OpPost2017, Affected by 2017 Swap and Owned By Postmedia or Metroland')
plt.xlabel('Year')
plt.ylabel('Circulation')
plt.title('Mean Circulation for Newspaper Editions of Each Year')
plt.legend()
plt.show()



grouped_data = FreeNewspaper.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='All Newspaper Editions Survive After 2017 In the Data')

Swap2017Affected=FreeNewspaper[FreeNewspaper['Swap2017Affected']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017 and Affected by Swap2017')

Swap2017Affected=FreeNewspaper[FreeNewspaper['OwnerPostmediaMetroland']==1]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017 and Owned By Postmedia or Metroland')

Swap2017Affected=FreeNewspaper[(FreeNewspaper['OwnerPostmediaMetroland']==1)&(FreeNewspaper['Swap2017Affected']==1)]
grouped_data=Swap2017Affected.groupby('Year')['NatlLineRate']
mean_rate = grouped_data.mean()
plt.plot(mean_circulation.index, mean_rate, 'o-', label='Newspapers Editions Survive After 2017, Affected by 2017 Swap and Owned By Postmedia or Metroland')

plt.xlabel('Year')
plt.ylabel('National Line Rate')
plt.title('Mean NatlLineRate for Newspaper Editions of Each Year')
plt.legend()
plt.show()




# Group the data by year and calculate statistics
NofNewspaperEdition = df.groupby('Year')['ID'].nunique()
plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='All Newspaper Editions In the Data')
Swap2017Affected=df[df['Swap2017Affected']==1]
NofNewspaperEdition = Swap2017Affected.groupby('Year')['ID'].nunique()
plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Affected by Swap2017')
OwnedByPM=df[df['OwnerPostmediaMetroland']==1]
NofNewspaperEdition = OwnedByPM.groupby('Year')['ID'].nunique()
plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Owned By Postmedia or Metroland')
OwnedByPM2017Affected=df[(df['OwnerPostmediaMetroland']==1)&(df['Swap2017Affected']==1)]
NofNewspaperEdition = OwnedByPM2017Affected.groupby('Year')['ID'].nunique()
plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Affected by 2017 Swap and Owned By Postmedia or Metroland')

plt.xlabel('Year')
plt.ylabel('Number of Newspaper Editions')
plt.title('Number for Newspaper Editions of Each Year')
plt.legend()
plt.show()







# Group the data by year and calculate statistics
#NofNewspaperEdition = CrossSubset.groupby('Year')['ID'].nunique()
#plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='All Newspaper Editions Survive After 2017 In the Data')
#Swap2017Affected=CrossSubset[CrossSubset['Swap2017Affected']==1]
#NofNewspaperEdition = Swap2017Affected.groupby('Year')['ID'].nunique()
#plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Survive After 2017 and Affected by Swap2017')
#OwnedByPM=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==1]
#NofNewspaperEdition = OwnedByPM.groupby('Year')['ID'].nunique()
#plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Survive After 2017 and Owned By Postmedia or Metroland')
#OwnedByPM2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==1)&(CrossSubset['Swap2017Affected']==1)]
#NofNewspaperEdition = OwnedByPM2017Affected.groupby('Year')['ID'].nunique()
#plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Survive After 2017, Affected by 2017 Swap and Owned By Postmedia or Metroland')


#NotOwnedByPM=CrossSubset[CrossSubset['OwnerPostmediaMetroland']==0]
#NofNewspaperEdition = NotOwnedByPM.groupby('Year')['ID'].nunique()
#plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Survive After 2017 and Not Owned By Postmedia or Metroland')
NotOwnedByPM2017Affected=CrossSubset[(CrossSubset['OwnerPostmediaMetroland']==0)&(CrossSubset['Swap2017Affected']==1)]
NofNewspaperEdition = NotOwnedByPM2017Affected.groupby('Year')['ID'].nunique()
plt.plot(NofNewspaperEdition.index, NofNewspaperEdition, 'o-', label='Newspapers Editions Survive After 2017, Affected by 2017 Swap and Not Owned By Postmedia or Metroland')



plt.xlabel('Year')
plt.ylabel('Number of Newspaper Editions')
plt.title('Number for Newspaper Editions of Each Year')
plt.legend()
plt.show()





paid_newspapers_to_drop = df[df['PaidCirculation'].notna() & (df['PaidCirculation'] > 0)]['ID'].unique()
FreeNewspapers = df[~df['ID'].isin(paid_newspapers_to_drop)]
FreeNewspapers.reset_index(drop=False, inplace=True)
FreeNewspapers.sort_values(by=['ID', 'Year'], inplace=True)
FreeNewspapers['ID'].nunique()


free_newspapers_to_drop = df[df['FreeCirculation'].notna() & (df['FreeCirculation'] > 0)]['ID'].unique()
PaidNewspapers = df[~df['ID'].isin(free_newspapers_to_drop)]
PaidNewspapers.reset_index(drop=False, inplace=True)
PaidNewspapers.sort_values(by=['ID', 'Year'], inplace=True)
PaidNewspapers['ID'] .nunique()




# Create an interaction term between 'Post2017' and 'OwnerPostmediaMetroland'
Mdf4Affected['Interaction'] = Mdf4Affected['Post2017']*Mdf4Affected['Swap2017Affected']*Mdf4Affected['OwnerPostmediaMetroland']
Mdf4Affected['Interaction2'] = Mdf4Affected['Post2017']*Mdf4Affected['Swap2017Affected']
Mdf4Affected['Interaction3'] = Mdf4Affected['OwnerPostmediaMetroland']*Mdf4Affected['Swap2017Affected']
# Set the owner, ID and Year as the index of the dataframe
Mdf4Affected.set_index(['CSDs', 'Year'], inplace=True)

Mdf4Affected.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CheckRegression.csv')
correlation_matrix = Mdf4Affected[['NatlLineRate','FreeCirculation']].corr()


# Run the fixed effects regression with owner, entity effects and interaction
model = PanelOLS.from_formula('NatlLineRate ~ Interaction+Interaction2', data=Mdf4Affected)
results=model.fit()

PostmediaMetroland=Mdf4Affected[Mdf4Affected['OwnerPostmediaMetroland']==1]
model = PanelOLS.from_formula('FreeCirculation ~ Interaction2+TimeEffects+EntityEffects', data=PostmediaMetroland)
results=model.fit()


Swap2017Affected=Mdf4Affected[Mdf4Affected['Swap2017Affected']==1]

model = PanelOLS.from_formula('FreeCirculation ~ Interaction+Time', data=Swap2017Affected)
results=model.fit()

Mdf4Affected.reset_index(['ID', 'Year'], inplace=True)

# Make sure your Year column is in the correct format
Mdf4Affected['Year'] = pd.to_datetime(Mdf4Affected['Year']).dt.year
grouped_means = Mdf4Affected.groupby(['Year', 'Swap2017Affected'])['FreeCirculation'].mean().reset_index()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='FreeCirculation', hue='Swap2017Affected', data=grouped_means)

plt.title('Mean Free Circulation by Year and Group')
plt.xlabel('Year')
plt.ylabel('Mean Free Circulation')
plt.show()

Swap2017Affected.reset_index(inplace=True)
Swap2017Affected['Year'] = pd.to_datetime(Swap2017Affected['Year']).dt.year
grouped_means = Swap2017Affected.groupby(['Year', 'Interaction3'])['FreeCirculation'].mean().reset_index()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='FreeCirculation', hue='Interaction3', data=grouped_means)

plt.title('Mean Free Circulation by Year and Group')
plt.xlabel('Year')
plt.ylabel('Mean Free Circulation')

plt.show()




# Define the groups you have in your dataset
groups = Mdf4Affected['Swap2017Affected'].unique()

# Define number of rows for subplot: one plot for each group
n = len(groups)

# Create a figure and axis array
fig, ax = plt.subplots(n, 1, figsize=(12, 8*n))  # Adjust the size (12, 8*n) as needed

# For each group, add a subplot
for i, group in enumerate(groups):
    group_data = grouped_means[grouped_means['Swap2017Affected'] == group]
    sns.lineplot(x='Year', y='FreeCirculation', data=group_data, ax=ax[i])
    ax[i].set_title(f'Mean Free Circulation by Year for {group}')
    ax[i].set_xlabel('Year')
    ax[i].set_ylabel('Mean Free Circulation')

# Display the plots
plt.tight_layout()
plt.show()





# extract coefficients
coeff_Post2017 = results.params['Post2017']
coeff_Interaction = results.params['Interaction']

# extract standard errors
se_Post2017 = results.std_errors['Post2017']
se_Interaction = results.std_errors['Interaction']

# compute test statistic
test_statistic = np.square(coeff_Post2017 - coeff_Interaction) / np.square(np.sqrt(np.square(se_Post2017) + np.square(se_Interaction)))

# compute p-value
p_value = 1 - stats.chi2.cdf(test_statistic, 1)

print('Test statistic:', test_statistic)
print('p-value:', p_value)




Mdf4Affected=Mdf4Affected[Mdf4Affected['FreeCirculation'].notna()]

# Group the data by year and extract the circulation values as a dictionary
data_by_year = Mdf4Affected.groupby('Year')['FreeCirculation'].apply(list).to_dict()

# Create a list to store the circulation data for each year
circulation_data = []

# Iterate over the years and append the circulation data to the list
for year, circulation in data_by_year.items():
    circulation_data.append(circulation)

# Plot the box plots
plt.boxplot(circulation_data)

# Set the x-axis labels as the years
plt.xticks(range(1, len(data_by_year) + 1), data_by_year.keys())

# Set the y-axis label
plt.ylabel('Circulation')

# Set the title of the plot
plt.title('Newspaper Circulation by Year')

# Display the plot
plt.show()

