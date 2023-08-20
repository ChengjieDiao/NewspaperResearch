import pandas as pd
import numpy as np
import re
import pickle
import ast
import json
import os

previous_directory = os.path.dirname(os.getcwd())

current_drive = os.path.splitdrive(os.getcwd())[0]
print(current_drive)

combined2=pd.read_csv(fr"{previous_directory}\CSD2016\combined2.csv",index_col=0,parse_dates=['Year'])

combined2['Year']=combined2['Year'].dt.year


combined2['CSDs']=combined2['CSDs'].apply(ast.literal_eval)

df=combined2.explode('CSDs')

CSDDemog16=pd.read_csv(fr"{previous_directory}\CSDRMapperData\CSDDemog16.csv")
CSDDemog16=CSDDemog16[CSDDemog16['Province name']=='ON']
CSDDemog16['CSD name']=CSDDemog16['CSD name'].str.strip()
CSDDemog16=CSDDemog16[~((CSDDemog16['CSD name']=='Hamilton') & (CSDDemog16['CD name']=='Northumberland'))]
##CSDDemog16.drop('Population and dwelling counts / Population percentage change, 2011 to 2016',axis=1,inplace=True)

start=CSDDemog16.columns.get_loc('Population and dwelling counts / Population, 2016')
end=CSDDemog16.columns.get_loc('Education / Total Sex / Total / Highest certificate, diploma or degree for the population aged 15 years and over in private households / 25% sample data / Postsecondary certificate, diploma or degree / University certificate, diploma or degree at bachelor level or above')

demographics = CSDDemog16.columns[start:end+1].tolist()
for col in CSDDemog16.columns[start:end+100]:
    CSDDemog16[col]=np.log(CSDDemog16[col]+100)


start2=df.columns.get_loc('Population and dwelling counts / Population, 2016')
end2=df.columns.get_loc('Immigration / Total Sex / Total / Immigrant status and period of immigration for the population in private households / 25% sample data / Immigrants')

df.drop(df.columns[start2:end2+1],axis=1,inplace=True)
df=pd.merge(df,CSDDemog16,how='left',left_on='CSDs',right_on='CSD name')
df.rename(columns={'Population and dwelling counts / Population, 2016':'Population','Population and dwelling counts / Population percentage change, 2011 to 2016':'PopulationGrowth','Age & Sex / Both sexes / Average age of the population ; Both sexes':'Age','Income / Total Sex / Total / Income statistics in 2015 for the population aged 15 years and over in private households / 100% data / Number of total income recipients aged 15 years and over in private households / 100% data / Median total income in 2015 among recipients ($)':'Income','Education / Total Sex / Total / Highest certificate, diploma or degree for the population aged 15 years and over in private households / 25% sample data / Postsecondary certificate, diploma or degree / University certificate, diploma or degree at bachelor level or above':'Education'},inplace=True)


df.sort_values(by=['CSDs','ID2','Year'],inplace=True)
df.drop_duplicates(subset=['CSDs','ID2','Year'],inplace=True)

df['NofID2'] = df.groupby(['CSDs', 'Year'])['ID2'].transform('nunique')

df.loc[df['EDITION'] == 'Monthly', ['FreeCirculation', 'TotalCirculation', 'PaidCirculation']] = df.loc[df['EDITION'] == 'Monthly', ['FreeCirculation', 'TotalCirculation', 'PaidCirculation']] / 4

# Unique values for 'CSDs', 'Year', and 'MetrolandOrPostmedia'
csds_unique = df['CSDs'].unique()
year_unique = df['Year'].unique()
metroland_unique = np.array([0, 1])  # assuming two unique values 0 and 1

# Create a multi-index with all combinations
multi_index = pd.MultiIndex.from_product([csds_unique, year_unique, metroland_unique],
                                         names=['CSDs', 'Year', 'MetrolandOrPostmedia'])

# Perform the groupby and sum operation
aggregate_circulation = df.groupby(['CSDs', 'Year', 'MetrolandOrPostmedia'])[['FreeCirculation', 'TotalCirculation', 'PaidCirculation']].sum()

# Reindex the DataFrame with the complete multi-index, filling missing groups with 0
aggregate_circulation = aggregate_circulation.reindex(multi_index, fill_value=0)







aggregate_circulation = aggregate_circulation.add_suffix('_Sum')

aggregate_circulation = aggregate_circulation.reset_index()


cols_to_suffix = ['FreeCirculation_Sum', 'TotalCirculation_Sum', 'PaidCirculation_Sum']
Circulation_Sum_Others=aggregate_circulation[aggregate_circulation['MetrolandOrPostmedia'] == 0].drop(columns='MetrolandOrPostmedia')
Circulation_Sum_Others.rename(columns={col: col + '_Others' for col in cols_to_suffix}, inplace=True)

Circulation_Sum_PM=aggregate_circulation[aggregate_circulation['MetrolandOrPostmedia'] == 1].drop(columns='MetrolandOrPostmedia')
Circulation_Sum_PM.rename(columns={col: col+'_PM' for col in cols_to_suffix},inplace=True)

df = df.merge(Circulation_Sum_Others, on=['CSDs', 'Year'], how='left')

# Merge the aggregated circulation data for PostmediaMetroland=1
df = df.merge(Circulation_Sum_PM,on=['CSDs', 'Year'], how='left')

#df[['FreeCirculation_Sum_Others', 'TotalCirculation_Sum_Others', 'PaidCirculation_Sum_Others','FreeCirculation_Sum_PM', 'TotalCirculation_Sum_PM', 'PaidCirculation_Sum_PM']].fillna(0)


df['PMShare']=(df['TotalCirculation_Sum_PM'])/(df['TotalCirculation_Sum_PM']+df['TotalCirculation_Sum_Others'])
df['PMShare'] = df.groupby(['CSDs', 'Year'])['PMShare'].apply(lambda group: group.interpolate(method='linear')).reset_index(level=['CSDs', 'Year'], drop=True)




# Create a temporary DataFrame with the counts of unique 'ID2' for each 'CSDs', 'Year' group where 'PostmediaMetroland' is 1
temp_df = df[df['MetrolandOrPostmedia'] == 1].groupby(['CSDs', 'Year'])['ID2'].nunique()
df['PMNofID2'] = df.set_index(['CSDs', 'Year']).index.map(temp_df.get).fillna(0)


df['MeanMarketRate'] = df.groupby(['CSDs', 'Year'])['NatlLineRateAllEds'].transform('mean')
temp_df = df[df['MetrolandOrPostmedia'] == 1].groupby(['CSDs', 'Year'])['NatlLineRateAllEds'].mean()
df['PMMeanMarketRate'] = df.set_index(['CSDs', 'Year']).index.map(temp_df.get)


df['MeanCirc']=df.groupby(['CSDs','Year'])['TotalCircAllEds'].transform('mean')



temp_df=df[df['MetrolandOrPostmedia']==1].groupby(['CSDs','Year'])['TotalCircAllEds'].mean()
df['PMMeanCirc']=df.set_index(['CSDs','Year']).index.map(temp_df.get)


# For 'MeanMarketRate', calculate the mean for years before 2017 for each 'CSDs'
df_before_2017 = df[df['Year'] < 2017]
mean_MeanMarketRate_before_2017 = df_before_2017.groupby('CSDs')['MeanMarketRate'].mean()
# For 'PMMeanMarketRate', calculate the mean for years before 2017 for each 'CSDs'
mean_PMMeanMarketRate_before_2017 = df_before_2017.groupby('CSDs')['PMMeanMarketRate'].mean()

# For 'NofID2', calculate the mean for years before 2017 for each 'CSDs'
mean_NofID2_before_2017 = df_before_2017.groupby('CSDs')['NofID2'].mean()

# Assume you also calculated 'PMNofID2' as you did 'NofID2'
# For 'PMNofID2', calculate the mean for years before 2017 for each 'CSDs'
mean_PMNofID2_before_2017 = df_before_2017.groupby('CSDs')['PMNofID2'].mean()


df['mean_MeanMarketRate_before_2017'] = df['CSDs'].map(mean_MeanMarketRate_before_2017)
df['mean_PMMeanMarketRate_before_2017'] = df['CSDs'].map(mean_PMMeanMarketRate_before_2017)
df['mean_NofID2_before_2017'] = df['CSDs'].map(mean_NofID2_before_2017)
df['mean_PMNofID2_before_2017'] = df['CSDs'].map(mean_PMNofID2_before_2017)


# For 'MeanMarketRate', calculate the mean for years after 2017 for each 'CSDs'
df_after_2017 = df[df['Year'] > 2017]
mean_MeanMarketRate_after_2017 = df_after_2017.groupby('CSDs')['MeanMarketRate'].mean()

# For 'PMMeanMarketRate', calculate the mean for years after 2017 for each 'CSDs'
mean_PMMeanMarketRate_after_2017 = df_after_2017.groupby('CSDs')['PMMeanMarketRate'].mean()

# For 'NofID2', calculate the mean for years after 2017 for each 'CSDs'
mean_NofID2_after_2017 = df_after_2017.groupby('CSDs')['NofID2'].mean()

# Assume you also calculated 'PMNofID2' as you did 'NofID2'
# For 'PMNofID2', calculate the mean for years after 2017 for each 'CSDs'
mean_PMNofID2_after_2017 = df_after_2017.groupby('CSDs')['PMNofID2'].mean()

# Map the calculated means to the corresponding 'CSDs' in the original dataframe
df['mean_MeanMarketRate_after_2017'] = df['CSDs'].map(mean_MeanMarketRate_after_2017)
df['mean_PMMeanMarketRate_after_2017'] = df['CSDs'].map(mean_PMMeanMarketRate_after_2017)
df['mean_NofID2_after_2017'] = df['CSDs'].map(mean_NofID2_after_2017)
df['mean_PMNofID2_after_2017'] = df['CSDs'].map(mean_PMNofID2_after_2017)




df.drop_duplicates(subset=['CSDs','Year'],inplace=True)

# Create a list of years from 2013 to 2019

df_pivot_mean = df.pivot(index='CSDs', columns='Year', values='MeanMarketRate').add_prefix('MeanMarketRate_').reset_index()
df_pivot_pmmean = df.pivot(index='CSDs', columns='Year', values='PMMeanMarketRate').add_prefix('PMMeanMarketRate_').reset_index()

# Merge the pivot dataframe back to the original dataframe
df= pd.merge(df, df_pivot_mean, on='CSDs', how='left')
df= pd.merge(df, df_pivot_pmmean, on='CSDs', how='left')

df.drop_duplicates(subset=['CSDs','Year'],inplace=True)

df.sort_values(by=['CSDs','Year'],inplace=True)

def calculate_time_trend(group,variable,trend):
    group = group.dropna(subset=[variable, 'Year'])
    group['Time']=group['Year']-group['Year'].where(group[variable].notna()).min()+1
    x=sm.add_constant(group['Time'])
    if len(group)>1:
        y=group[variable]
        model=sm.OLS(y,x)
        results=model.fit()
        group[f'{variable}_{trend}']=results.params['Time']
    else:
        group[f'{variable}_{trend}']=np.nan
    return group

from functools import partial
import statsmodels.api as sm


partial_func_pre_PM = partial(calculate_time_trend, variable='PMMeanMarketRate', trend='PreTrend')
partial_func_post_PM = partial(calculate_time_trend, variable='PMMeanMarketRate', trend='PostTrend')

partial_func_pre = partial(calculate_time_trend, variable='MeanMarketRate', trend='PreTrend')
partial_func_post = partial(calculate_time_trend, variable='MeanMarketRate', trend='PostTrend')



results=df[df['Year'].between(2013,2017)].groupby('CSDs').apply(partial_func_pre_PM).reset_index(drop=True).drop_duplicates(subset=['CSDs']).drop(columns='Year')
df = df.merge(results[['CSDs', 'PMMeanMarketRate_PreTrend']], on=['CSDs'], how='left')
results=Rate_Trend_2013_2017=df[df['Year'].between(2013,2017)].groupby('CSDs').apply(partial_func_pre).reset_index(drop=True).drop_duplicates(subset=['CSDs']).drop(columns='Year')
df=df.merge(results[['CSDs','MeanMarketRate_PreTrend']],on=['CSDs'],how='left')
results=df[df['Year'].between(2017,2019)].groupby('CSDs').apply(partial_func_post_PM).reset_index(drop=True).drop_duplicates(subset=['CSDs']).drop(columns='Year')
df=df.merge(results[['CSDs','PMMeanMarketRate_PostTrend']],on=['CSDs'],how='left').reset_index(drop=True)
results=Rate_Trend_2013_2017=df[df['Year'].between(2017,2019)].groupby('CSDs').apply(partial_func_post).reset_index(drop=True).drop_duplicates(subset=['CSDs']).drop(columns='Year')
df=df.merge(results[['CSDs','MeanMarketRate_PostTrend']],on=['CSDs'],how='left')


with open(fr"{previous_directory}\CSD2016\affected_market.pkl", 'rb') as file:
    TMarket = pickle.load(file)


df['TMarket']=0
df.loc[df['CSDs'].isin(TMarket),'TMarket']=1

print(len(df))

df[df['TMarket']==1].to_csv(r'Check2.csv')

Treatment=df[df['TMarket']==1].drop_duplicates(['CSDs','Year'])
Control=df[df['TMarket']==0].drop_duplicates(['CSDs','Year'])

df_filtered=Treatment[Treatment['TMarket']==1].sort_values(by=['CSDs','Year'])
df_filtered.to_csv(r'Check.csv')


print("Number of Treatment Market",Treatment['CSDs'].nunique())
print("Number of Controal Market",Control['CSDs'].nunique())




ProbitDf=df[df['Year']==2017]



ProbitDf.drop_duplicates(subset='CSDs',inplace=True)

ProbitDf.to_csv(r'Check.csv')

ProbitDf['MeanCirc']=np.log(ProbitDf['MeanCirc']+100)
ProbitDf['PMMeanCirc']=np.log(ProbitDf['PMMeanCirc']+100)



ProbitDf.columns = ProbitDf.columns.str.lower()

#ProbitDf=ProbitDf[ProbitDf['pmnofid2']>0]

## Run Probit Model

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.iolib.summary2 import summary_col


formula1 = 'TMarket ~ Population + PopulationGrowth + Age + Income + Education + NofID2'
formula2 = 'TMarket ~ Population + PopulationGrowth + Age + Income + Education + NofID2 + MeanMarketRate_PreTrend + MeanMarketRate_2017 + MeanCirc'
formula3 = 'TMarket ~ Population + PopulationGrowth + Age + Income + Education + NofID2 + PMNofID2 + MeanMarketRate_PreTrend + PMMeanMarketRate_PreTrend + MeanMarketRate_2017 + PMMeanMarketRate_2017 + PMShare + MeanCirc + PMMeanCirc'



ProbitDf1=ProbitDf[['tmarket','population','populationgrowth','age','income','education','nofid2']]
ProbitDf2=ProbitDf[['tmarket','population','populationgrowth','age','income','education','nofid2','meanmarketrate_pretrend','meanmarketrate_2017','meancirc']]
ProbitDf3=ProbitDf[['tmarket','population','populationgrowth','age','income','education','nofid2','pmnofid2','meanmarketrate_pretrend','pmmeanmarketrate_pretrend','meanmarketrate_2017','pmmeanmarketrate_2017','pmshare','meancirc','pmmeancirc']]


ProbitDf1.dropna(inplace=True)
ProbitDf2.dropna(inplace=True)
ProbitDf3.dropna(inplace=True)
# Convert formulas to lowercase
formula1 = formula1.lower()
formula2 = formula2.lower()
formula3 = formula3.lower()
# Reorder your dataframe columns based on the formula
variables1 = formula1.split("~")[1].strip().split(" + ")
variables2 = formula2.split("~")[1].strip().split(" + ")
variables3 = formula3.split("~")[1].strip().split(" + ")
columns_ordered1 = variables1
columns_ordered2 = variables2
columns_ordered3 = variables3

#ProbitDf1 = ProbitDf[columns_ordered1]
#ProbitDf2 = ProbitDf[columns_ordered2]



ProbitDf = ProbitDf.dropna()
# Fit the models
probit_model1 = smf.probit(formula=formula1, data=ProbitDf1).fit()
probit_model2 = smf.probit(formula=formula2, data=ProbitDf2).fit()
probit_model3 = smf.probit(formula=formula3, data=ProbitDf3).fit()

# Print the model summary
print(probit_model1.summary())
print(probit_model2.summary())
print(probit_model3.summary())
# Create a list of fitted models



models = [probit_model1, probit_model2, probit_model3]
orders = [columns_ordered1, columns_ordered2, columns_ordered3]



# Call summary_col to combine results
summary_table = summary_col(models, stars=True, float_format='%0.4f',
                  model_names=['probit_model1', 'probit_model2','probit_model3'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'Pseudo R2':lambda x: "{:.2f}".format(x.prsquared)})

summary_string = summary_table.as_text().split('\n')

# Find index where results table starts and ends
start_index = next(index for index, value in enumerate(summary_string) if value.startswith('===================='))
end_index = next(index for index, value in enumerate(summary_string) if value.startswith('------------------'))

# Slice the summary_string to keep only the results table
summary_list = summary_string[(start_index+1):end_index]

# Split each row in summary_list into a list of values
summary_list = [row.split() for row in summary_list]

# Convert to DataFrame
summary_df = pd.DataFrame(summary_list[1:], columns=summary_list[0])

# Export to CSV
summary_df.to_csv('summary_table.csv', index=False)









# First, set 'CSDs' and 'Year' as the multi-index
df.set_index(['CSDs', 'Year'], inplace=True)

# Drop rows with missing advertising rates
df.dropna(subset=['MeanMarketRate'], inplace=True)
# Define the set of all required years
required_years = set(range(2015, 2019))

# Create a list of all CSDs that have records for all required years
valid_csd = [csd for csd in df.index.get_level_values('CSDs').unique() if required_years.issubset(set(df.loc[csd].index.get_level_values('Year')))]

# Keep only rows where CSDs is in valid_csd
df = df[df.index.get_level_values('CSDs').isin(valid_csd)]

# Reset index
df.reset_index(inplace=True)


Treatment=df[df['TMarket']==1].drop_duplicates(['CSDs','Year'])
Control=df[df['TMarket']==0].drop_duplicates(['CSDs','Year'])

print("Number of Treatment Market 2",Treatment['CSDs'].nunique())
print("Number of Controal Market 2",Control['CSDs'].nunique())


# First, set 'CSDs' and 'Year' as the multi-index
df.set_index(['CSDs', 'Year'], inplace=True)

# Drop rows with missing advertising rates
df.dropna(subset=['MeanMarketRate'], inplace=True)

# Define the set of all required years
required_years = set(range(2013, 2020))

# Create a list of all CSDs that have records for all required years
valid_csd = [csd for csd in df.index.get_level_values('CSDs').unique() if required_years.issubset(set(df.loc[csd].index.get_level_values('Year')))]

# Keep only rows where CSDs is in valid_csd
df = df[df.index.get_level_values('CSDs').isin(valid_csd)]

# Reset index
df.reset_index(inplace=True)


Treatment=df[df['TMarket']==1].drop_duplicates(['CSDs','Year'])
Control=df[df['TMarket']==0].drop_duplicates(['CSDs','Year'])

print("Number of Treatment Market 3",Treatment['CSDs'].nunique())
print("Number of Controal Market 3",Control['CSDs'].nunique())

