import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from scipy import stats
import ast
import pickle

import os


previous_directory = os.path.dirname(os.getcwd())

import matplotlib
import matplotlib.pyplot as plt
# Assuming that 'owner' is a column in your data


df = pd.read_csv(fr"{previous_directory}\CSD2016\Combined.csv", parse_dates=['Year'])





postmedia=df[df['Owner']=='postmedia network inc.'];

postmedia.to_csv(r"E:\IOnewspaper\openaipdf\CSDRMapperData\postmedia.csv")

cols_to_drop = df.columns[df.columns.str.startswith('Unnamed')]
df.drop(columns=cols_to_drop,inplace=True)


with open(fr"{previous_directory}\CSD2016\affected_market.pkl", 'rb') as f:
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




MAndPAffectedAvailible=df[(df['MetrolandOrPostmedia']==1)&(df['OpCross2017ID2']==1)&(df['Rate_Cross17_ID2']==1)&(df['Swap2017Affected']==1)]
MAndPAffectedAvailible.sort_values(by=['ID2','Year'],inplace=True)
MAndPAffectedAvailible['ID2'].nunique()
MAndPControlAvailible=df[(df['MetrolandOrPostmedia']==1)&(df['OpCross2017ID2']==1)&(df['Rate_Cross17_ID2']==1)&(df['Swap2017Affected']==0)]
MAndPControlAvailible['ID2'].nunique()
MAndPAffectedAvailible.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\MAndPAffectedAvailible.csv')
MAndPControlAvailible.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\MAndPControlAvailible.csv')



df.sort_values(by=['ID2','Year'],inplace=True)
df.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\Combined2.csv')
### postmedia matching  ###



PMatchCsd=df[(df['postmedia']==1)&(df['OpCross2017ID2']==1)&(df['Rate_Cross17_ID2']==1)]




PMatchCsd.sort_values(by=['ID2','Year'],inplace=True)

PMatchCsd.to_csv(r'Check.csv')

## subset the data with only need columns ##

PMatchCsd.sort_values(by=['ID2','Year'],inplace=True)
PMatchCsd2=PMatchCsd[['ID2','MarketAndNewspaper','Owner','Year','NatlLineRateAllEds','Swap2017','Swap2017Affected','CSDs']]

## drop duplicates for ID2 and Year, as we only need to know each newspaper rate, Not for each Edition


PMatchCsd2.to_csv(fr"{previous_directory}\CSDRMapperData\checkPMatchCsd.csv")


PMatchCsd2 = PMatchCsd2.drop_duplicates(subset=['ID2', 'Year'])

PMatchCsdAffected2=PMatchCsd2[PMatchCsd2['Swap2017Affected']==1]

PMatchCsdNotAffected2=PMatchCsd2[PMatchCsd2['Swap2017Affected']==0]






# Filter for years 2018 and 2019 and calculate the average
avg_ad_rate_2018_2019 = PMatchCsd2[PMatchCsd2['Year'].dt.year.isin([2018, 2019])].groupby('ID2')['NatlLineRateAllEds'].mean()

# Filter for year 2017
ad_rate_2017 = PMatchCsd2[PMatchCsd2['Year'].dt.year == 2017].groupby('ID2')['NatlLineRateAllEds'].mean()
adg_ad_rate_2013_2016= PMatchCsd2[PMatchCsd2['Year'].dt.year <= 2016].groupby('ID2')['NatlLineRateAllEds'].mean()

# Subtract the 2017 rate from the average of 2018 and 2019 rates
difference1 = avg_ad_rate_2018_2019 - ad_rate_2017

# Convert the difference Series to a DataFrame
difference_df1 = difference1.reset_index()

# Rename the column in the difference DataFrame
difference_df1.columns = ['ID2', 'RateDifference2017_20182019']

# Merge the difference DataFrame back into the original DataFrame
PMatchCsd2 = pd.merge(PMatchCsd2, difference_df1, on='ID2', how='left')
PMatchCsd2 = PMatchCsd2.drop_duplicates(subset=['ID2'])

difference2 = ad_rate_2017 - adg_ad_rate_2013_2016
difference_df2 = difference2.reset_index()
difference_df2.columns = ['ID2', 'RateDifference_20132016_2017']
PMatchCsd2 = pd.merge(PMatchCsd2, difference_df2, on='ID2', how='left')


PMatchCsd2.sort_values(by=['Swap2017Affected','ID2','Year'],inplace=True)
PMatchCsd2.to_csv(fr"{previous_directory}\CSDRMapperData\checkPMatchCsdMeanDifference.csv")



mean_rate_difference = PMatchCsd2.groupby('Swap2017Affected')[['RateDifference_20132016_2017','RateDifference2017_20182019']].mean()
mean_rate_difference.to_csv(fr"{previous_directory}\CSDRMapperData\checkPMatchCsdGroupMeanDifference.csv")


## calculate the mean_rate_difference for metroland ##




MMatchCsd=df[(df['metroland']==1)&(df['OpCross2017ID2']==1)&(df['Rate_Cross17_ID2']==1)]

MMatchCsd.sort_values(by=['ID2','Year'],inplace=True)



## subset the data with only need columns ##

MMatchCsd.sort_values(by=['ID2','Year'],inplace=True)
MMatchCsd2=MMatchCsd[['ID2','MarketAndNewspaper','Owner','Year','NatlLineRateAllEds','Swap2017','Swap2017Affected','CSDs']]

## drop duplicates for ID2 and Year, as we only need to know each newspaper rate, Not for each Edition


MMatchCsd2.to_csv(fr"{previous_directory}\CSDRMapperData\checkMMatchCsd.csv")


MMatchCsd2 = MMatchCsd2.drop_duplicates(subset=['ID2', 'Year'])

MMatchCsdAffected2=MMatchCsd2[MMatchCsd2['Swap2017Affected']==1]
MMatchCsdNotAffected2=MMatchCsd2[MMatchCsd2['Swap2017Affected']==0]




# Filter for years 2018 and 2019 and calculate the average
avg_ad_rate_2018_2019 = MMatchCsd2[MMatchCsd2['Year'].dt.year.isin([2018, 2019])].groupby('ID2')['NatlLineRateAllEds'].mean()
# Filter for year 2017
ad_rate_2017 = MMatchCsd2[MMatchCsd2['Year'].dt.year == 2017].groupby('ID2')['NatlLineRateAllEds'].mean()
adg_ad_rate_2013_2016= MMatchCsd2[MMatchCsd2['Year'].dt.year <= 2016].groupby('ID2')['NatlLineRateAllEds'].mean()

# Subtract the 2017 rate from the average of 2018 and 2019 rates
difference1 = avg_ad_rate_2018_2019 - ad_rate_2017

# Convert the difference Series to a DataFrame
difference_df1 = difference1.reset_index()

# Rename the column in the difference DataFrame
difference_df1.columns = ['ID2', 'RateDifference2017_20182019']

# Merge the difference DataFrame back into the original DataFrame
MMatchCsd2 = pd.merge(MMatchCsd2, difference_df1, on='ID2', how='left')
MMatchCsd2 = MMatchCsd2.drop_duplicates(subset=['ID2'])

difference2 = ad_rate_2017 - adg_ad_rate_2013_2016
difference_df2 = difference2.reset_index()
difference_df2.columns = ['ID2', 'RateDifference_20132016_2017']
MMatchCsd2 = pd.merge(MMatchCsd2, difference_df2, on='ID2', how='left')


MMatchCsd2.sort_values(by=['Swap2017Affected','ID2','Year'],inplace=True)
MMatchCsd2.to_csv(fr"{previous_directory}\CSDRMapperData\checkMMatchCsdMeanDifference.csv")
mean_rate_difference = MMatchCsd2.groupby('Swap2017Affected')[['RateDifference_20132016_2017','RateDifference2017_20182019']].mean()



mean_rate_difference.to_csv(fr"{previous_directory}\CSDRMapperData\checkMMatchCsdGroupMeanDifference.csv")





Mintersection= set()
Pintersection = set()
# Iterate over each row (list) in the DataFrame
for index, row in PMatchCsdAffected2.iterrows():
    Pintersection.update(row['CSDs'])

for index, row in MMatchCsdAffected2.iterrows():
    Mintersection.update(row['CSDs'])


with open(fr"{previous_directory}\CSDRMapperData\Pintersection.pkl", 'wb') as f:
    # Dump the list to the file
    pickle.dump(Pintersection, f)


with open(fr"{previous_directory}\CSDRMapperData\Mintersection.pkl", 'wb') as f:
    # Dump the list to the file
    pickle.dump(Mintersection, f)

unique_PMatchCsd = set()
# Iterate over each row (list) in the DataFrame
for index, row in PMatchCsd.iterrows():
    unique_PMatchCsd.update(row['CSDs'])


unique_PMatchCsd.difference_update(affected_market)
print(unique_PMatchCsd)


with open(fr"{previous_directory}\CSDRMapperData\unique_PMatchCsd.pkl", 'wb') as f:
    # Dump the list to the file
    pickle.dump(unique_PMatchCsd, f)


### metroland matching  ###

MMatchCsd=df[(df['metroland']==1) & (df['OpCross2017ID2']==1) & (df['Rate_Cross17_ID2']==1)]
MMatchCsd.sort_values(by=['ID2','Year'],inplace=True)
MMatchCsd2=MMatchCsd[['MarketAndNewspaper','Owner','Year','NatlLineRate','NatlLineRateAllEds','Swap2017','Swap2017Affected','CSDs']]
MMatchCsd2.to_csv(fr"{previous_directory}\CSDRMapperData\checkMMatchCsd.csv")


unique_MMatchCsd = set()

# Iterate over each row (list) in the DataFrame
for index, row in MMatchCsd.iterrows():
    unique_MMatchCsd.update(row['CSDs'])


unique_MMatchCsd.difference_update(affected_market)
print(unique_MMatchCsd)


with open(fr"{previous_directory}\CSDRMapperData\unique_MMatchCsd.pkl", 'wb') as f:
    # Dump the list to the file
    pickle.dump(unique_MMatchCsd, f)

PMatchCsd_set = set(unique_PMatchCsd)
MMatchCsd_set = set(unique_MMatchCsd)

import pandas as pd

# Assuming df is your dataframe and 'CSD_List' is your column with lists of CSDs
# Also assuming provided_CSD_list is your provided list of CSDs

# Create an empty list to store the results
dfs = pd.DataFrame()

# Loop over each csd in your provided list
for csd in PMatchCsd_set:
    # Get a boolean mask where each value is True if csd is in the list for that row, False otherwise
    mask = df['CSDs'].apply(lambda x: csd in x)

    # Subset the dataframe using this mask
    subset_df = df[mask].copy()

    # Add a new column with the current csd
    subset_df['CSD'] = csd

    # Append the resulting DataFrame to the list
    dfs=pd.concat([dfs,subset_df],ignore_index=True)
dfs.sort_values(by=['CSD','Year','Owner','ID2'],inplace=True)
dfs.to_csv(fr"{previous_directory}\CSDRMapperData\PMatchCsd_Data.csv")
# Now result_df is a dataframe containing all rows where 'CSD_List' contains any csd in provided_CSD_list, with an additional column 'CSD' indicating the CSD

dfs=pd.DataFrame();
for csd in MMatchCsd_set:
    mask=df['CSDs'].apply(lambda x: csd in x)
    subset_df=df[mask].copy();
    subset_df['CSD']=csd;

    dfs=pd.concat([dfs,subset_df],ignore_index=True)

dfs.sort_values(by=['CSD','Year','Owner','ID2'],inplace=True)
dfs.to_csv(fr"{previous_directory}\CSDRMapperData\MMatchCsd_Data.csv")
# Use the apply function to check if any CSDs in the list for each row are in provided_CSD_set
dfPMatchCsd_set = df[df['CSDs'].apply(lambda x: any(csd in PMatchCsd_set for csd in x))]
dfPMatchCsd_set.to_csv(fr"{previous_directory}\CSDRMapperData\dfPMatchCsd_set.csv")




# Use the apply function to check if any CSDs in the list for each row are in provided_CSD_set
dfMMatchCsd_set = df[df['CSDs'].apply(lambda x: any(csd in PMatchCsd_set for csd in x))]
dfMMatchCsd_set.to_csv(fr"{previous_directory}\CSDRMapperData\dfMMatchCsd_set.csv")

CSDDemog16=pd.read_csv(fr"{previous_directory}\CSDRMapperData\CSDDemog16.csv")
CSDDemog16=CSDDemog16[CSDDemog16['Province name']=='ON']
CSDDemog16['CSD name']=CSDDemog16['CSD name'].str.strip()
CSDDemog16.drop('Population and dwelling counts / Population percentage change, 2011 to 2016',axis=1,inplace=True)

start=CSDDemog16.columns.get_loc('Population and dwelling counts / Population, 2016')
end=CSDDemog16.columns.get_loc('Education / Total Sex / Total / Highest certificate, diploma or degree for the population aged 15 years and over in private households / 25% sample data / Postsecondary certificate, diploma or degree / University certificate, diploma or degree at bachelor level or above')

demographics = CSDDemog16.columns[start:end+1].tolist()
for col in CSDDemog16.columns[start:end+1]:
    CSDDemog16[col]=np.log(CSDDemog16[col]+1)
CSDDemog16.to_csv(fr"{previous_directory}\CSDRMapperData\check.csv")


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the demographic data in df1 and transform both data sets
data_scaled = scaler.fit_transform(CSDDemog16[demographics])
# Replace the original demographic data with the scaled data
CSDDemog16[demographics] = data_scaled
affected_market_Demog16 = CSDDemog16[CSDDemog16['CSD name'].isin(affected_market)]
unique_MMatchCsd_Demog16 = CSDDemog16[CSDDemog16['CSD name'].isin(unique_MMatchCsd)]
unique_PMatchCsd_Demog16 = CSDDemog16[CSDDemog16['CSD name'].isin(unique_PMatchCsd)]
unique_MMatchCsd_Demog16 = CSDDemog16[CSDDemog16['CSD name'].isin(unique_MMatchCsd)]
unique_MMatchCsd_Demog16=unique_MMatchCsd_Demog16[~((unique_MMatchCsd_Demog16['CSD name']=='Hamilton') & (unique_MMatchCsd_Demog16['CD name']=='Northumberland'))]


with open(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Pintersection.pkl', 'rb') as file:
    Pintersection = pickle.load(file)
with open(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Mintersection.pkl', 'rb') as file:
    Mintersection = pickle.load(file)




Pintersection_Demog16=affected_market_Demog16[affected_market_Demog16['CSD name'].isin(Pintersection)]
Mintersection_Demog16=affected_market_Demog16[affected_market_Demog16['CSD name'].isin(Mintersection)]



from sklearn.metrics import pairwise_distances_argmin_min

## postmedia matching the adv rate
closest_indices, _ = pairwise_distances_argmin_min(Pintersection_Demog16[demographics], unique_PMatchCsd_Demog16[demographics])
# Map each CSD in Pintersection to the most similar CSD in unique_PMatchCsd_Demog16
Pintersection_Demog16['closest_CSD'] = unique_PMatchCsd_Demog16.iloc[closest_indices]['CSD name'].values
Pintersection_Demog16.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Pintersection_Demog16_Matched_Adv.csv')

## metroland matching the adv rate
closest_indices, _ = pairwise_distances_argmin_min(Mintersection_Demog16[demographics], unique_MMatchCsd_Demog16[demographics])
# Map each CSD in Pintersection to the most similar CSD in unique_PMatchCsd_Demog16
Mintersection_Demog16['closest_CSD'] = unique_MMatchCsd_Demog16.iloc[closest_indices]['CSD name'].values
Mintersection_Demog16.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\Mintersection_Demog16_Matched_Adv.csv')


df = pd.read_csv(fr"{previous_directory}\CSD2016\Combined2.csv", parse_dates=['Year'])



df['Year']=df['Year'].dt.year

postmediadf=df[df['postmedia']==1]
metrolanddf=df[df['metroland']==1]

## DID for adversting rate for Postmedia in each market:


import numpy as np
import pandas as pd

# Assuming your data is in a DataFrame df with columns 'ID2', 'CSDs', 'Year' and 'Advertising Rate'

# Function to calculate average difference in advertising rate for a given ID2
#def calculate_difference(df):
#    early_years = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]['NatlLineRateAllEds'].mean()
#    later_years = df[(df['Year'] >= 2018) & (df['Year'] <= 2019)]['NatlLineRateAllEds'].mean()
#    return later_years - early_years

def calculate_difference1(df):
    early_years = df[(df['Year']==2017)]['NatlLineRateAllEds'].mean()
    later_years = df[df['Year']>2017]['NatlLineRateAllEds'].mean()
    return later_years - early_years


def calculate_difference2(df):
    later_years = df[(df['Year']==2017)]['NatlLineRateAllEds'].mean()
    early_years = df[df['Year']<2017]['NatlLineRateAllEds'].mean()
    return later_years - early_years


# Apply the function to each ID2
postmediadf_grouped1 = postmediadf.groupby('ID2').apply(calculate_difference1).reset_index()


postmediadf_grouped1.columns = ['ID2', 'Average_Difference_Adv_2017_20182019']

postmediadf_grouped1.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\postmediadf_grouped.csv')

postmediadf_grouped2 = postmediadf.groupby('ID2').apply(calculate_difference2).reset_index()


postmediadf_grouped2.columns = ['ID2', 'Average_Difference_Adv_20132016_2017']

postmediadf_grouped2.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\postmediadf_grouped2.csv')

# Merge df_grouped with the original df to propagate the Average_Difference to each row
postmediadf = postmediadf.merge(postmediadf_grouped1, on='ID2', how='left')
postmediadf = postmediadf.merge(postmediadf_grouped2, on='ID2', how='left')

# Create a DataFrame to store the mean average differences for each CSD
mean_difference_per_csd_Postmedia = pd.DataFrame(columns=['CSD', 'Mean_Average_Difference_Adv_20132016_2017','Mean_Average_Difference_Adv_2017_20182019','closest_CSD','Mean_Average_Difference_Adv_20132016_2017_closest_CSD','Mean_Average_Difference_Adv_2017_20182019_closest_CSD'])

# Calculate the mean average difference for each CSD in Pintersection
# Calculate the mean average difference for each CSD in Pintersection
for index, row in Pintersection_Demog16.iterrows():
    csd = row['CSD name']
    csdMatch = row['closest_CSD']
    df_filtered = postmediadf[postmediadf['CSDs'].apply(lambda x: csd in x)]
    df_filteredM = postmediadf[postmediadf['CSDs'].apply(lambda x: csdMatch in x)]

    # Calculate and append the mean average difference
    temp_df = pd.DataFrame({
        'CSD': [csd],
        'Mean_Average_Difference_Adv_2017_20182019': [df_filtered['Average_Difference_Adv_2017_20182019'].mean()],
        'Mean_Average_Difference_Adv_20132016_2017': [df_filtered['Average_Difference_Adv_20132016_2017'].mean()],
        'closest_CSD': [csdMatch], 'Mean_Average_Difference_Adv_2017_20182019_closest_CSD': [
            df_filteredM['Average_Difference_Adv_2017_20182019'].mean()],
        'Mean_Average_Difference_Adv_20132016_2017_closest_CSD': [
            df_filteredM['Average_Difference_Adv_20132016_2017'].mean()]
    })

    mean_difference_per_csd_Postmedia = pd.concat([mean_difference_per_csd_Postmedia, temp_df], ignore_index=True)







# Calculate the mean of each numeric column
# Calculate the mean of each numeric column
mean_values = mean_difference_per_csd_Postmedia.mean(numeric_only=True)

# For non-numeric columns add specific value, e.g., None
non_numeric = mean_difference_per_csd_Postmedia.select_dtypes(exclude=[np.number]).columns
non_numeric_series = pd.Series([None]*len(non_numeric), index=non_numeric)

# Concatenate mean values and non numeric series
mean_values = pd.concat([mean_values, non_numeric_series])

mean_values_df = pd.DataFrame(mean_values).T
mean_difference_per_csd_Postmedia =pd.concat([mean_difference_per_csd_Postmedia,mean_values_df], ignore_index=True)

print(mean_difference_per_csd_Postmedia)

mean_difference_per_csd_Postmedia.to_csv("E:\IOnewspaper\openaipdf\CSDRMapperData\PostmediaMarketMatched.csv")

## DID for adversting rate for Metroland in each market:


import numpy as np
import pandas as pd

# Apply the function to each ID2
metrolanddf_grouped1 = metrolanddf.groupby('ID2').apply(calculate_difference1).reset_index()

metrolanddf_grouped1.columns = ['ID2', 'Average_Difference_Adv_2017_20182019']

metrolanddf_grouped1.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\metrolanddf_grouped.csv')

metrolanddf_grouped2 = metrolanddf.groupby('ID2').apply(calculate_difference2).reset_index()

metrolanddf_grouped2.columns = ['ID2', 'Average_Difference_Adv_20132016_2017']

metrolanddf_grouped2.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\metrolanddf_grouped2.csv')

# Merge df_grouped with the original df to propagate the Average_Difference to each row
metrolanddf = metrolanddf.merge(metrolanddf_grouped1, on='ID2', how='left')
metrolanddf = metrolanddf.merge(metrolanddf_grouped2, on='ID2', how='left')

# Create a DataFrame to store the mean average differences for each CSD
mean_difference_per_csd_Metroland = pd.DataFrame(columns=['CSD', 'Mean_Average_Difference_Adv_20132016_2017','Mean_Average_Difference_Adv_2017_20182019','closest_CSD','Mean_Average_Difference_Adv_20132016_2017_closest_CSD','Mean_Average_Difference_Adv_2017_20182019_closest_CSD'])

# Calculate the mean average difference for each CSD in Pintersection
# Calculate the mean average difference for each CSD in Pintersection
for index, row in Mintersection_Demog16.iterrows():
    csd = row['CSD name']
    csdMatch = row['closest_CSD']
    df_filtered = metrolanddf[metrolanddf['CSDs'].apply(lambda x: csd in x)]
    df_filteredM = metrolanddf[metrolanddf['CSDs'].apply(lambda x: csdMatch in x)]

    # Calculate and append the mean average difference
    # Calculate and append the mean average difference
    # Calculate and append the mean average difference
    temp_df = pd.DataFrame({
        'CSD': [csd],
        'Mean_Average_Difference_Adv_2017_20182019': [df_filtered['Average_Difference_Adv_2017_20182019'].mean()],'Mean_Average_Difference_Adv_20132016_2017': [df_filtered['Average_Difference_Adv_20132016_2017'].mean()],'closest_CSD':[csdMatch],'Mean_Average_Difference_Adv_2017_20182019_closest_CSD':[df_filteredM['Average_Difference_Adv_2017_20182019'].mean()],'Mean_Average_Difference_Adv_20132016_2017_closest_CSD':[df_filteredM['Average_Difference_Adv_20132016_2017'].mean()]
    })

    mean_difference_per_csd_Metroland = pd.concat([mean_difference_per_csd_Metroland, temp_df], ignore_index=True)

# Calculate the mean of each numeric column
# Calculate the mean of each numeric column
mean_values = mean_difference_per_csd_Metroland.mean(numeric_only=True)

# For non-numeric columns add specific value, e.g., None
non_numeric = mean_difference_per_csd_Metroland.select_dtypes(exclude=[np.number]).columns
non_numeric_series = pd.Series([None]*len(non_numeric), index=non_numeric)

# Concatenate mean values and non numeric series
mean_values = pd.concat([mean_values, non_numeric_series])

mean_values_df = pd.DataFrame(mean_values).T
mean_difference_per_csd_Metroland =pd.concat([mean_difference_per_csd_Metroland,mean_values_df], ignore_index=True)

print(mean_difference_per_csd_Metroland)

mean_difference_per_csd_Metroland.to_csv("E:\IOnewspaper\openaipdf\CSDRMapperData\MetrolandMarketMatched.csv")

unique_MMatchCsd_Demog16.sort_values(by=['CSD name'],inplace=True)

