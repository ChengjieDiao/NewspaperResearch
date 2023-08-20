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

df4=pd.read_csv(fr"{previous_directory}\CSD2016\CSDdf4-2-2.csv",index_col=0)

# Replace single quotes with double quotes
df4['CSDs'] = df4['CSDs'].str.replace('[\'Cochrane, Unorganized, North Part\']','[\'Cochrane\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Kenora, Unorganized\']','[\'Kenora\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Rainy River, Unorganized\']','[\'Rainy River\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Thunder Bay, Unorganized\']','[\'Thunder Bay\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Greater Napanee\', \'Stone Mills\', \'Brudenell, Lyndoch and Raglan\']','[\'Greater Napanee\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Haldimand County\', \'Six Nations (Part) 40\']','[\'Haldimand County\']')
df4['CSDs'] = df4['CSDs'].str.replace('[\'Six Nations (Part) 40\']','[\'Haldimand County\']')

# Remove 'Unorganized' from the lists

df4.to_csv('check.csv')

df4['Newspaper'] = df4['Newspaper'].str.replace(' */ *', '/', regex=True).str.lower()
df4['Newspaper'] = df4['Newspaper'].str.replace('??','')
df4.loc[df4['Newspaper']=='kanata kourierstandard','Newspaper']='kanata kourier standard'
df4.loc[df4['Newspaper']=='standard guideadvocate','Newspaper']='standard guide advocate'

StandardizeCSDdf4NewspaperName1=pd.read_csv(fr"{previous_directory}\CSD2016\StandardizeCSDdf4NewspaperName1.csv")
StandardizeCSDdf4NewspaperName1['Newspaper']=StandardizeCSDdf4NewspaperName1['Newspaper'].str.replace(' */ *', '/', regex=True).str.strip().str.lower()
StandardizeCSDdf4NewspaperName1['Newspaper']=StandardizeCSDdf4NewspaperName1['Newspaper'].str.replace('??','')


Mdf4 = pd.merge(df4, StandardizeCSDdf4NewspaperName1, on=['Newspaper','Market','Owner'], how='left')
Mdf4.loc[Mdf4['StdName2'].notnull(), 'Newspaper'] = Mdf4['StdName2']
Mdf4.loc[Mdf4['StdOwner2'].notnull(), 'Owner'] = Mdf4['StdOwner2']
Mdf4.loc[Mdf4['StdMarket2'].notnull(),'Market']=Mdf4['StdMarket2']
Mdf4['Newspaper'] = Mdf4['Newspaper'].str.lower().str.replace('/',' ').str.replace('&','').str.strip()
Mdf4.loc[Mdf4['EDITION']=='12-times per year','EDITION']='Monthly'
Mdf4.sort_values(by=['Newspaper','EDITION', 'Year'], inplace=True)  # sort by newspaper and year
Mdf4['MarketChange1']=Mdf4['Market'].str.replace('%20',' ').str.replace('&',' ').str.replace('/',' ')
Mdf4['Daily']=0;
Mdf4['SwapNotInclude']=0;



#Mdf4[Mdf4['_merge']!='both'].to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\Check.csv')

CircUnincluded=pd.read_csv(fr"{previous_directory}\CSD2016\CircNotIncludedNewspaperAdd.csv", dtype={'CSDs': str})
# Convert 'CSDs' column from string to a list of elements
CircUnincluded['CSDs']=CircUnincluded['CSDs'].str.strip()
CircUnincluded['Year'] = CircUnincluded['Year'].str.strip("[]").str.split(',')
CircUnincluded = CircUnincluded.explode('Year')
CircUnincluded['Year'] = CircUnincluded['Year'].astype(int)
CircUnincluded.reset_index(drop=True, inplace=True)
CircUnincluded.loc[CircUnincluded['Year']>2017,'Owner']=CircUnincluded['OwnerAfterSwap']
CircUnincluded.loc[CircUnincluded['Daily']!=1,'Daily']=0
CircUnincluded.drop(['OwnerAfterSwap'],axis=1,inplace=True)
CircUnincluded['SwapNotInclude']=1


missing_columns = [col for col in Mdf4.columns if col not in CircUnincluded.columns]
# Set the missing columns in CircUnincluded to appropriate default values based on the column dtype in Mdf4
for col in missing_columns:
    if Mdf4[col].dtype == 'object':  # 'object' dtype usually means strings in pandas
        CircUnincluded[col] = [''] * len(CircUnincluded)
    elif Mdf4[col].dtype in ['int', 'int32', 'int64']:
        CircUnincluded[col] = np.full(len(CircUnincluded), 0, dtype=Mdf4[col].dtype)
    else:  # for 'float', 'float32', 'float64' dtypes
        CircUnincluded[col] = np.full(len(CircUnincluded), np.nan, dtype=Mdf4[col].dtype)



Mdf4 = pd.concat([Mdf4, CircUnincluded], ignore_index=True)
Mdf4.to_csv(fr"{previous_directory}\CSD2016\check2.csv")




CircUnincluded.to_csv(fr"{previous_directory}\CSD2016\check.csv")

def create_market_and_newspaper(row):
    market_words = set(re.split(r'\s+|/', row['MarketChange1'].lower()))
    newspaper_words = set(re.split(r'\s+|/', row['Newspaper'].lower()))
    common_words = market_words.intersection(newspaper_words)
    ##print(common_words)
    if 'the' in common_words and len(common_words) == 1:
        return row['MarketChange1'] + ' ' + row['Newspaper']
    elif len(common_words) >=1:
        return row['Newspaper']
    else:
        return row['MarketChange1'] + ' ' + row['Newspaper']

Mdf4['MarketAndNewspaper'] = Mdf4.apply(create_market_and_newspaper, axis=1)
Mdf4.loc[Mdf4['MarketChange1']=='','MarketAndNewspaper']=Mdf4.loc[Mdf4['MarketChange1']=='','Market'].to_numpy() +' '+ Mdf4.loc[Mdf4['MarketChange1']=='','Newspaper'].to_numpy()
Mdf4.to_csv(fr"{previous_directory}\CSD2016\check.csv")




Mdf4['MarketAndNewspaper'] = Mdf4['MarketAndNewspaper'].replace('\s+', ' ', regex=True).str.strip()
Mdf4=Mdf4[Mdf4['MarketAndNewspaper']!='wasaga sun']
Mdf4=Mdf4[Mdf4['MarketAndNewspaper']!='stayner the sun']
Mdf4.sort_values(by=['MarketAndNewspaper','EDITION', 'Year'], inplace=True)  # sort by newspaper and year


import ast

def remove_duplicates(string_list):
    list_repr = ast.literal_eval(string_list)
    unique_list = list(set(list_repr))
    return str(unique_list)

Mdf4['Unique_CSDs'] = Mdf4['CSDs'].apply(remove_duplicates)
Mdf4.drop('CSDs',axis=1,inplace=True)
Mdf4.rename(columns={'Unique_CSDs':'CSDs'},inplace=True)
# Convert 'CSDs' column from string to a list of elements
Mdf4['CSDs'] = Mdf4['CSDs'].apply(
    lambda x: [element.strip(" '\"") for element in x.strip("[]").split(",")])
# Sort the DataFrame by Newspaper, EDITION, and Year and reset the index
Mdf4 = Mdf4.sort_values(['Newspaper', 'EDITION', 'Year']).reset_index(drop=True)
# Create a new column 'ID' to store the unique identity number
Mdf4['ID'] = ''
Mdf4['ID2'] = ''
# Initialize a counter for assigning unique IDs
counter = 1
counter2 = 1
# Iterate over the rows of the DataFrame
for index, row in Mdf4.iterrows():
    newspaper = row['Newspaper']
    edition = row['EDITION']
    csds = set(row['CSDs'])
    market=row['Market']
    # Check if the current row matches the previous row based on Newspaper and EDITION
    if index > 0 and newspaper == Mdf4.at[index - 1, 'Newspaper'] and edition == Mdf4.at[index - 1, 'EDITION']:
        if market != Mdf4.at[index-1,'Market']:
        # Check if the CSDs intersection is not empty with the previous row
            if csds.intersection(set(Mdf4.at[index - 1, 'CSDs'])):
            # Assign the same ID as the previous row
                Mdf4.at[index, 'ID'] = Mdf4.at[index - 1, 'ID']
            else:
            # Assign a new unique ID
                Mdf4.at[index, 'ID'] = counter
                counter += 1
        else:
            Mdf4.at[index, 'ID'] = Mdf4.at[index - 1, 'ID']
    else:
        # Assign a new unique ID
        Mdf4.at[index, 'ID'] = counter
        counter += 1

    if index > 0 and newspaper == Mdf4.at[index - 1, 'Newspaper']:
        if market != Mdf4.at[index - 1, 'Market']:
            # Check if the CSDs intersection is not empty with the previous row
            if csds.intersection(set(Mdf4.at[index - 1, 'CSDs'])):
                # Assign the same ID as the previous row
                Mdf4.at[index, 'ID2'] = Mdf4.at[index - 1, 'ID2']
            else:
                # Assign a new unique ID
                Mdf4.at[index, 'ID2'] = counter2
                counter2 += 1
        else:
            Mdf4.at[index, 'ID2'] = Mdf4.at[index - 1, 'ID2']
    else:
        # Assign a new unique ID
        Mdf4.at[index, 'ID2'] = counter2
        counter2 += 1



Mdf4.sort_values(by=['ID','Year'],inplace=True)

Mdf4.to_csv(fr"{previous_directory}\CSD2016\check.csv")


### if curent circulation equals to previous one, set it to zero###

unique_ids_1 = Mdf4['ID'].unique()
Mdf4['first_year'] = Mdf4.groupby(['ID'])['Year'].transform('min')  # get the first year for each newspaper
Mdf4['equal_to_previous'] = Mdf4.groupby(['ID'])['FreeCirculation'].shift() == Mdf4['FreeCirculation']
Mdf4['equal_to_previous'] = Mdf4['equal_to_previous'] & (Mdf4['FreeCirculation'] != 0)  # check if current circulation equals to the previous one
Mdf4.loc[(Mdf4['Year'] != Mdf4['first_year']) & Mdf4['equal_to_previous'], 'FreeCirculation'] = np.nan
Mdf4['equal_to_previous'] = Mdf4.groupby(['ID'])['PaidCirculation'].shift() == Mdf4['PaidCirculation']  # check if current circulation equals to the previous one
Mdf4['equal_to_previous'] = Mdf4['equal_to_previous'] & (Mdf4['PaidCirculation'] != 0)
Mdf4.loc[(Mdf4['Year'] != Mdf4['first_year']) & Mdf4['equal_to_previous'], 'PaidCirculation'] = np.nan
Mdf4['equal_to_previous'] = Mdf4.groupby(['ID'])['TotalCirculation'].shift() == Mdf4['TotalCirculation']  # check if current circulation equals to the previous one
Mdf4['equal_to_previous'] = Mdf4['equal_to_previous'] & (Mdf4['TotalCirculation'] != 0)
Mdf4.loc[(Mdf4['Year'] != Mdf4['first_year']) & Mdf4['equal_to_previous'], 'TotalCirculation'] = np.nan




unique_ids_1=Mdf4['ID'].unique().tolist()
#MergeACSubCol=MergeAdvAndCirc[['ID','Year','FreeCirculation','TotalCirculation','PaidCirculation','Swap2017','Swap2017Affected','Owner','AdvCirc','NatlLineRate','SpotColour','ProcessColour','CSDs']
Mdf4.sort_values(by=['ID','Year'],inplace=True)
#### study the Merged Dataset, Include All Merged Newspapers ##############
columns_to_exclude = ['ID','Year']  # List of columns to exclude

for id in unique_ids_1:
    id_data = Mdf4[Mdf4['ID'] == id]
    min_year, max_year = id_data['Year'].min(), id_data['Year'].max()
    for year in range(min_year + 1, max_year):
        if year not in id_data['Year'].values:
            new_row_data = {'ID': [id], 'Year': [year]}
            for column in Mdf4.columns:
                if column not in columns_to_exclude:
                    new_row_data[column] = [np.nan]
            new_row = pd.DataFrame(new_row_data)
            Mdf4 = pd.concat([Mdf4, new_row], ignore_index=True)



# Define the condition for each group (i.e., each newspaper)

linear_interpolation_cols = ['FreeCirculation', 'TotalCirculation', 'PaidCirculation']
other_cols = [col for col in Mdf4.columns if col not in linear_interpolation_cols]



def interpolate_column(df, column):
    return df[column].interpolate(method='linear',limit_area='inside')

Mdf4.sort_values(by=['ID', 'Year'], inplace=True)

for column in linear_interpolation_cols:
    Mdf4[column] = Mdf4.groupby('ID').apply(interpolate_column, column=column).reset_index(level=0, drop=True)

Mdf4.sort_values(by=['ID', 'Year'], inplace=True)
Mdf4[other_cols] = Mdf4.groupby('ID')[other_cols].ffill()
Mdf4.drop(columns=Mdf4.columns[Mdf4.columns.str.contains('Unnamed')], inplace=True)
Mdf4['TotalCirculation']=Mdf4['FreeCirculation']+Mdf4['PaidCirculation']

# Calculate mean for different IDs within each ID2 and year combination
mean_cir = Mdf4.groupby(['ID2', 'Year']).agg({'ID': 'nunique', 'FreeCirculation': 'mean', 'PaidCirculation': 'mean','TotalCirculation': 'mean'}).reset_index()
mean_cir.rename(columns={'FreeCirculation':'FreeCircAllEds','PaidCirculation':'PaidCircAllEds','TotalCirculation':'TotalCircAllEds'},inplace=True)
mean_cir.drop('ID', axis=1,inplace=True)
Mdf4=pd.merge(Mdf4,mean_cir,how='left',on=['ID2','Year'])



Mdf4.to_csv(fr"{previous_directory}\CSD2016\check_1.csv",index=False)



# Print the updated DataFrame
num_unique_ids = Mdf4['ID'].nunique()
print("Number of unique ID: ", num_unique_ids)
## Sort How many Ids"





NCSDs=set(Mdf4.explode('CSDs')['CSDs'])

dfSwappedMarketAllYears=Mdf4[(Mdf4['Swap2017']==1)]

dfSwappedMarketAllYears.sort_values(by=['MarketAndNewspaper','Year'],inplace=True)
dfSwappedMarketAllYears[['MarketAndNewspaper','CSDs','Market','Year']].to_csv(fr"{previous_directory}\CSD2016\dfSwappedMarketAllYears.csv")

dfSwappedMarket=Mdf4[(Mdf4['Swap2017']==1) & (Mdf4['Year'].isin([2013,2014,2014,2016,2017,2018,2019]))]
dfSwappedMarket.sort_values(by=['Newspaper','Year'],inplace=True)
dfSwappedMarket.to_csv(fr"{previous_directory}\CSD2016\checkSwapped.csv")

ListSwappedNewspaper = dfSwappedMarket.sort_values('MarketAndNewspaper')['MarketAndNewspaper'].unique().tolist()

ListSwappedNewspaper.sort(key=str.lower)


with open(fr"{previous_directory}\CSD2016\output.txt", 'w') as file:
    # Write each element of the list to the file
    for item in ListSwappedNewspaper:
        file.write(str(item) + '\n')


print('List Of Swapped Newspapers')
print(ListSwappedNewspaper)
print(len(ListSwappedNewspaper))
SwappedMarket=set(dfSwappedMarket.explode('CSDs')['CSDs'])
print('Number Of Affected CSDs')
print(len(SwappedMarket))



with open(fr"{previous_directory}\CSD2016\\affected_market.pkl", 'wb') as file:
    # Serialize and write the list to the file
    pickle.dump(SwappedMarket, file)

def check_element(element_list, element_set):
    for element in element_list:
        if element in element_set:
            return 1
    return 0

# Apply the function to update the "Swap2017Affected" column
Mdf4['Swap2017Affected'] = Mdf4['CSDs'].apply(lambda x: check_element(x, SwappedMarket))
Mdf4.loc[Mdf4['Swap2017'].isnull(),'Swap2017']=0
Mdf4.to_csv(fr"{previous_directory}\CSD2016\Mdf4CSDsorted.csv")




####### Doing Analysis For the Advertising rate Data ##########

# Custom date parsing function
def date_parser(date_str):
    try:
        # Try parsing the date using the first format
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        try:
            # Try parsing the date using the second format
            return pd.to_datetime(date_str, format='%m/%d/%y')
        except ValueError:
            # Handle any other format here or return null/NaN if necessary
            return pd.NaT

advs1=pd.DataFrame()
for i in range(2013, 2020):
    advs=pd.read_csv(fr'{current_drive}\IOnewspaper\\advertising rate\{i}ad.csv',parse_dates=['AuditDate'], date_parser=date_parser)
    advs['Year']=i
    advs1=pd.concat([advs1,advs],axis=0)

advs1['AuditYear']=advs1['AuditDate'].dt.year.fillna(0).astype(int)
advs1['AuditDiff']=advs1['Year']-advs1['AuditYear']

advs1.loc[advs1['AuditDiff']>1,['Circ']]=np.nan

advs1.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\checkadvs1.csv')
advs1.to_csv(fr'{current_drive}\IOnewspaper\\advertising rate\advs1.csv')


advs1['Newspaper']=advs1['Newspaper'].str.replace('-MO|-TU|-WE|-TH|-FR|-SA|-SU| - NEW|,', '', regex=True).str.replace('-',' ').str.replace('NewsNow','News Now')
advs1['Newspaper']=advs1['Newspaper'].str.replace('Ind. Free Press', 'independent & free press', regex=True).str.replace('.This Wk',' This Week').str.replace('Econ.','Economist').str.replace('Ind. F.P.','independent & free press').str.replace('ClaringtonThis Wk','Clarington This Week')
advs1['Newspaper']=advs1['Newspaper'].str.replace(r'\(.*?\)', '', regex=True).str.replace(r'E.','East ').str.replace(r'Expr.','Express')
advs1['Newspaper']=advs1['Newspaper'].str.replace(r'/Weekender', ' Weekender', regex=True).str.replace('Pt.','Port').str.replace('L\'Express','express l').str.replace('Osh/','Oshawa/').str.replace('Whit/','Whiteby/').str.replace('/Clar','/Clarington')
advs1['Newspaper']=advs1['Newspaper'].str.lower().replace('st. lawrence emc','st. lawrence news').str.replace(r'\s{2,}', ' ', regex=True).str.replace(r'belleville emc','belleville news',regex=True)
advs1['Newspaper']=advs1['Newspaper'].str.replace('frontenac emc','frontenac news').str.replace('kanata kourier standard emc','kanata kourier standard').str.replace('arnprior chronicle guide emc','arnprior chronicle guide')
advs1['Newspaper']=advs1['Newspaper'].str.replace('nepean/barrhaven emc','nepean/barrhaven news')
advs1['Newspaper']=advs1['Newspaper'].str.replace('manotick emc','manotick news').str.replace('orleans emc','orleans news').str.replace('ottawa east emc','ottawa east news').str.replace('ottawa west emc','ottawa west news').str.replace('ottawa south emc','ottawa south news')
advs1['Newspaper']=advs1['Newspaper'].str.replace('quinte west emc','quinte west news').str.replace('renfrew mercury emc','renfrew mercury').str.replace('stittsville news emc','stittsville news').str.replace('west carleton review emc','west carleton review').str.replace('port perry scugog standard','port perry/uxbridge the standard').str.replace('port perry/uxbridge scugog standard','port perry/uxbridge the standard')
advs1['Newspaper']=advs1['Newspaper'].str.replace('/',' ')
advs1['Newspaper']=advs1['Newspaper'].str.replace('&',' ')
advs1['Newspaper']=advs1['Newspaper'].str.replace('whiteby','whitby')
advs1.dropna(subset=['Newspaper'], inplace=True)
advs1['Newspaper'] = advs1['Newspaper'].replace('\s+', ' ', regex=True).str.strip()
advs1.loc[advs1['Newspaper']=='chatham kent this week','Newspaper']='chatham this week'
advs1.columns=advs1.columns.str.strip()
for col in ['Circ','NatlLineRate','SpotColour','ProcessColour']:
    advs1[col] = advs1[col].str.replace('$', '').str.replace(',', '')
    advs1[col] = pd.to_numeric(advs1[col], errors='coerce')



StandardizeAdv1NameManual=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\StandardizeAdv1NameManual.csv',index_col=0)
advs1 = pd.merge(advs1, StandardizeAdv1NameManual, on=['Newspaper'], how='left')
advs1.loc[advs1['stdName2'].notnull(),'Newspaper']=advs1['stdName2']
advs1.drop('stdName2',axis=1,inplace=True)
advs1.reset_index(drop=True,inplace=True)
advs1.sort_values(by=['Newspaper','Year'],ascending=[True,True],inplace=True)
advs1['Issued']=advs1['Issued'].str.replace('1st/3rd Friday','Friday').str.replace('EO Friday','Friday').str.replace('18x/year','Monthly').str.replace('17x/Year','Monthly').str.replace('EO Thur','Thursday').str.replace('EO Wed','Wednesday')
advs1['Issued']=advs1['Issued'].str.replace('M/Tu/W/F','Monday/Tuesday/Wednesday/Friday').str.replace('W/Th/F','Wednesday/Thursday/Friday').str.replace('Thursdaysday','Thursday').str.replace('Mon/Wed','Monday/Wednesday').str.replace('Mon/Wed','Monda/Wednesday').str.replace('Tue/Fri','Tuesday/Friday').str.replace('2nd/4th Wed','Wednesday').str.replace('Wed/Thur','Wednesday/Thursday').str.replace('M/W/F','Monday/Wednesday/Friday').str.replace('Thur/Sat','Thursday/Saturday').str.replace('Thur/Fri','Thursday/Friday').str.strip()
advs1['SplitIssue']=advs1['Issued'].str.split('/')
advs1=advs1.explode('SplitIssue')
advs1.reset_index(drop=True,inplace=True)
advs1=advs1.rename(columns={'Circ':'AdvCirc','Newspaper':'AdvNewspaper'})
advs1.sort_values(by=['AdvNewspaper','SplitIssue','Year'],inplace=True)
advs1.reset_index(drop=True,inplace=True)
advs1=advs1.drop(columns=['Audit','AuditDate','Issued','Format']).rename(columns={'Circ':'AdvCirc','Newspaper':'AdvNewspaper'})



advs1['ID']=''
advs1['ID2']=''
counter=1;
counter2=1;

for index, row in advs1.iterrows():
    AdvNewspaper = row['AdvNewspaper']
    SplitIssue = row['SplitIssue']
    # Check if the current row matches the previous row based on Newspaper and SplitIssue
    if index > 0 and ((AdvNewspaper == advs1.at[index - 1, 'AdvNewspaper']) and (SplitIssue == advs1.at[index - 1, 'SplitIssue'])):
        advs1.at[index, 'ID'] = advs1.at[index - 1, 'ID']
    else:
        advs1.at[index, 'ID'] = counter
        counter += 1
    # Check if the current row matches the previous row based on Newspaper and SplitIssue
    if index > 0 and (AdvNewspaper == advs1.at[index - 1, 'AdvNewspaper']):
        advs1.at[index, 'ID2'] = advs1.at[index - 1, 'ID2']
    else:
        advs1.at[index, 'ID2'] = counter2
        counter2 += 1




linear_interpolation_cols = ['AdvCirc',	'NatlLineRate',	'SpotColour','ProcessColour']
other_cols = [col for col in advs1.columns if col not in linear_interpolation_cols]
advs1.sort_values(by=['ID', 'Year'], inplace=True)

for column in linear_interpolation_cols:
    advs1[column] = advs1.groupby('ID').apply(interpolate_column, column=column).reset_index(level=0, drop=True)

advs1.sort_values(by=['ID', 'Year'], inplace=True)
advs1[other_cols] = advs1.groupby('ID')[other_cols].ffill()
advs1.drop(columns=advs1.columns[advs1.columns.str.contains('Unnamed')], inplace=True)


advs1.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\checkadv1.csv')
# Calculate mean for different IDs within each ID2 and year combination
mean_cir = advs1.groupby(['ID2', 'Year']).agg({'ID': 'nunique', 'AdvCirc': 'mean', 'NatlLineRate': 'mean','SpotColour': 'mean','ProcessColour':'mean'}).reset_index()
mean_cir.rename(columns={'AdvCirc':'AdvCircAllEds','NatlLineRate':'NatlLineRateAllEds','SpotColour':'SpotColourAllEds','ProcessColour':'ProcessColourAllEds'},inplace=True)
mean_cir.drop('ID', axis=1,inplace=True)
advs1=pd.merge(advs1,mean_cir,how='left',on=['ID2','Year'])
advs1.rename(columns={'ID':'AdvID','ID2':'AdvID2'},inplace=True)





FuzzyMatchAdvCircManual=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\FuzzyMatchAdvCircManual.csv')
FuzzyMatchAdvCircManual.loc[FuzzyMatchAdvCircManual['correct'].notnull(),'MarketAndNewspaper']=FuzzyMatchAdvCircManual['correct']
FuzzyMatchAdvCircManual.drop(FuzzyMatchAdvCircManual[(FuzzyMatchAdvCircManual['wrong']==1)&FuzzyMatchAdvCircManual['correct'].isnull()].index,inplace=True)
FuzzyMatchAdvCircManual.drop(columns=['wrong','correct'],inplace=True)
MergeAdvAndCirc=pd.merge(advs1, FuzzyMatchAdvCircManual, on=['AdvNewspaper', 'AdvNewspaper'], how='outer',indicator=True)
MergeAdvAndCirc.rename(columns={'_merge':'_merge1'},inplace=True)


MergeAdvAndCirc.loc[MergeAdvAndCirc['_merge1']=='left_only','MarketAndNewspaper']=MergeAdvAndCirc['AdvNewspaper']
AdvOnlyManual=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\AdvOnlyManual.csv')
MergeAdvAndCirc=pd.merge(MergeAdvAndCirc,AdvOnlyManual,on=['AdvNewspaper'],how='left')
MergeAdvAndCirc=MergeAdvAndCirc[MergeAdvAndCirc['Removable']!=1]


MergeAdvAndCirc['ADVOwner']=MergeAdvAndCirc['ADVOwner'].str.strip();

#MergeAdvAndCirc.loc[MergeAdvAndCirc['ADVOwner'].notnull(),'Owner']=MergeAdvAndCirc['ADVOwner']

MergeAdvAndCirc[MergeAdvAndCirc['ADVOwner'].notnull()].to_csv("check.csv")

AdvOnly=pd.DataFrame(MergeAdvAndCirc[MergeAdvAndCirc['_merge1']=='left_only']['AdvNewspaper'].unique(),columns=['AdvNewspaper'])


AdvOnly.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\AdvOnly.csv',index=False)


MergeAdvAndCirc.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\check_advs1.csv',index=False)


MergeAdvAndCirc['EDITION']=MergeAdvAndCirc['SplitIssue']

Mdf4=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Mdf4CSDsorted.csv')


Mdf4['EDITION2']=Mdf4['EDITION']
# Assume df1 and df2 are your dataframes and 'A', 'B', 'C' are your columns
# First try a full merge
advMerge=MergeAdvAndCirc
MergeAdvAndCirc = pd.merge(advMerge, Mdf4, how='outer', on=['EDITION', 'Year', 'MarketAndNewspaper'], indicator=True)

# Identify rows with left_only and right_only
left_only = MergeAdvAndCirc[MergeAdvAndCirc['_merge'] == 'left_only']

right_only = MergeAdvAndCirc[MergeAdvAndCirc['_merge'] == 'right_only']

MergeAdvAndCirc.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\CheckCombined1.csv')
left_only.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Checkleft_only1.csv')
right_only.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Checkright_only1.csv')
# For these rows, try to merge on 'A' and 'B'
# Since 'A' and 'B' are not unique in df2 and df1, you'll first have to define a rule for selecting a match
# Here we're just going to group by 'A' and 'B' in df2 and df1 and select the first row of each group
Mdf4_grouped = Mdf4.groupby(['MarketAndNewspaper', 'Year']).first().reset_index()
advMerge_grouped = advMerge.groupby(['MarketAndNewspaper', 'Year']).first().reset_index()

# imperfect match for left_only and right_only
left_only = pd.merge(left_only, Mdf4_grouped, how='left', on=['MarketAndNewspaper', 'Year'], suffixes=('', '_y'), indicator='imperfect_merge')
right_only = pd.merge(right_only,advMerge_grouped, how='left', on=['MarketAndNewspaper', 'Year'], suffixes=('', '_y'), indicator='imperfect_merge')

# For columns in df2, fill NA values in left_only with values from the imperfect match
for column in Mdf4_grouped.columns:
    if column not in ['MarketAndNewspaper', 'Year','EDITION']:
        left_only[column] = left_only[column].fillna(left_only[f'{column}_y'])
        left_only = left_only.drop(columns=[f'{column}_y'])

left_only = left_only.drop(columns=['EDITION_y'])

# For columns in df1, fill NA values in right_only with values from the imperfect match
for column in advMerge_grouped.columns:
    if column not in ['MarketAndNewspaper', 'Year','EDITION']:
        right_only[column] = right_only[column].fillna(right_only[f'{column}_y'])
        right_only = right_only.drop(columns=[f'{column}_y'])

right_only = right_only.drop(columns=['EDITION_y'])


# Indicate these as imperfect matches
left_only['imperfect_merge'] = np.where(left_only['imperfect_merge']=='both', 'left_imperfect_match', 'left_only')
left_only['_merge']=left_only['imperfect_merge']
right_only['imperfect_merge'] = np.where(right_only['imperfect_merge']=='both', 'right_imperfect_match', 'right_only')
right_only['_merge']=right_only['imperfect_merge']

left_only.drop('imperfect_merge',axis=1,inplace=True)
right_only.drop('imperfect_merge',axis=1,inplace=True)

# Now we need to combine the perfect matches and the imperfect matches
perfect_match = MergeAdvAndCirc[MergeAdvAndCirc['_merge'] == 'both']

perfect_match.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\CheckPerfectMatch.csv')

right_only.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\CheckRightOnly.csv')
combined = pd.concat([perfect_match, left_only, right_only])

# Clean up the merge indicators
combined['_merge'] = np.where(combined['_merge']=='both', 'perfect_match', combined['_merge'])
combined.rename(columns={'_merge':'_merge2'},inplace=True)


combined.sort_values(by=['MarketAndNewspaper','Year','EDITION'],inplace=True)

left_only_merge1=combined[combined['_merge1']=='left_only']
right_only_merge1=combined[combined['_merge1']=='right_only']
left_only_merge1.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\left_only_merge1.csv')
right_only_merge1.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\right_only_merge1.csv')

left_only_merge2=combined[combined['_merge2']=='left_only']
right_only_merge2=combined[combined['_merge2']=='right_only']
left_only_merge2.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\left_only_merge2.csv')
right_only_merge2.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\right_only_merge2.csv')

combined.drop(['ID','ID2','AdvID','AdvID2'],axis=1,inplace=True)

combined.sort_values(by=['MarketAndNewspaper','EDITION','Year'],inplace=True)
combined.reset_index(drop=True,inplace=True)



# Create a new column 'ID' to store the unique identity number
combined['ID'] = ''
combined['ID2'] = ''
# Initialize a counter for assigning unique IDs
counter = 1
counter2 = 1

combined.loc[combined['SwapNotInclude']==1,'EDITION']=''
# Iterate over the rows of the DataFrame
for index, row in combined.iterrows():
    MarketAndNewspaper = row['MarketAndNewspaper']
    edition = row['EDITION']
    # Check if the current row matches the previous row based on Newspaper and EDITION
    if (index > 0) and ((MarketAndNewspaper == combined.at[index - 1, 'MarketAndNewspaper']) and (edition == combined.at[index - 1, 'EDITION'])):
        combined.at[index, 'ID'] = combined.at[index - 1, 'ID']
    else:
        combined.at[index, 'ID'] = counter
        counter += 1

    if (index > 0 )and (MarketAndNewspaper == combined.at[index - 1, 'MarketAndNewspaper']):
        combined.at[index, 'ID2'] = combined.at[index - 1, 'ID2']
    else:
        combined.at[index, 'ID2'] = counter2
        counter2 += 1


combined.sort_values(by=['MarketAndNewspaper','EDITION','Year'],inplace=True)
combined['ADVCSD'] = combined['ADVCSD'].apply(lambda x: [x])
combined.reset_index(inplace=True,drop=True)


combined.loc[(combined['_merge1']=='left_only')|((combined['_merge2']=='left_only')&(combined['_merge1']!='both')),'CSDs']=combined['ADVCSD']
combined.loc[(combined['_merge2']=='left_imperfect_match'),['FreeCirculation','PaidCirculation','TotalCirculation']]=combined.loc[(combined['_merge2']=='left_imperfect_match'),['FreeCircAllEds','PaidCircAllEds','TotalCircAllEds']].to_numpy()
combined.loc[(combined['_merge2']=='right_imperfect_match'),['AdvCirc','NatlLineRate','SpotColour','ProcessColour']]=combined.loc[(combined['_merge2']=='right_imperfect_match'),['AdvCircAllEds','NatlLineRateAllEds','SpotColourAllEds','ProcessColourAllEds']].to_numpy()
excluded_columns=['FreeCirculation','PaidCirculation','TotalCirculation','FreeCircAllEds','PaidCircAllEds','TotalCircAllEds','AdvCirc','NatlLineRate','SpotColour','ProcessColour','AdvCircAllEds','NatlLineRateAllEds','SpotColourAllEds','ProcessColourAllEds','Lowner','EDITION']




combined.sort_values(by=['ID','Year'],inplace=True)
combined.reset_index(drop=True,inplace=True)
other_cols = [col for col in combined.columns if col not in excluded_columns]
combined[other_cols] = combined.groupby('ID')[other_cols].ffill()
combined[other_cols] = combined.groupby('ID')[other_cols].bfill()
combined[other_cols] = combined.groupby('ID2')[other_cols].ffill()
combined[other_cols] = combined.groupby('ID2')[other_cols].bfill()

combined.sort_values(by=['ID2','Year'],inplace=True);


for column in excluded_columns:
    combined[column] = combined.groupby('ID2').apply(interpolate_column, column=column).reset_index(level=0, drop=True)





combined.loc[combined['Swap2017'].isnull(),'Swap2017']=0




PaidNewspaper = combined[combined['PaidCirculation'].notna() & (combined['PaidCirculation'] > 0)]['ID'].unique().tolist()
combined['EverPaidEDITION']=0;
combined.loc[combined['ID'].isin(PaidNewspaper),'EverPaidEDITION']=1
FreeNewspaper = combined[combined['FreeCirculation'].notna() & (combined['FreeCirculation'] > 0)]['ID'].unique().tolist()
combined['EverFreeEDITION']=0;
combined.loc[combined['ID'].isin(FreeNewspaper),'EverFreeEDITION']=1
PaidNewspaper = combined[combined['PaidCirculation'].notna() & (combined['PaidCirculation'] > 0)]['ID'].unique().tolist()
combined['EverPaid']=0;
combined.loc[combined['ID'].isin(PaidNewspaper),'EverPaid']=1
FreeNewspaper = combined[combined['FreeCirculation'].notna() & (combined['FreeCirculation'] > 0)]['ID'].unique().tolist()
combined['EverFree']=0;
combined.loc[combined['ID'].isin(FreeNewspaper),'EverFree']=1



def condition(df):
    pre2018 = df[df['Year'] <= 2017]
    post2017 = df[df['Year'] > 2017]
    return (not pre2018.empty) and (not post2017.empty)


OpCross2017ID = combined.groupby('ID').apply(condition).reset_index(name='OpCross2017ID')
# Get IDs of newspapers to keep
OpCross2017ID = OpCross2017ID[OpCross2017ID['OpCross2017ID']]['ID'].unique().tolist()


combined['OpCross2017ID']=0
# Filter the DataFrame
combined.loc[combined['ID'].isin(OpCross2017ID),'OpCross2017ID']=1


OpCross2017ID2 = combined.groupby('ID2').apply(condition).reset_index(name='OpCross2017ID2')
# Get IDs of newspapers to keep
OpCross2017ID2 = OpCross2017ID2[OpCross2017ID2['OpCross2017ID2']]['ID2'].unique().tolist()


combined['OpCross2017ID2']=0
# Filter the DataFrame
combined.loc[combined['ID2'].isin(OpCross2017ID2),'OpCross2017ID2']=1



def check_condition_FreeCirc_Cross17_ID2(group):
    # Check the years
    years = [2017, 2018, 2019]
    for year in years:
        # If 'FreeCirculation' is null for any of the specified years, return False
        year_subset = group[group['Year'] == year];

        if year_subset.empty or year_subset['FreeCircAllEds'].isnull().any() or (year_subset['FreeCircAllEds'] == 0).any():
            return False
    # If 'FreeCirculation' is not null for all specified years, return True
    return True

def check_condition_PaidCirc_Cross17_ID2(group):
    # Check the years
    years = [2017, 2018, 2019]
    for year in years:
        # If 'FreeCirculation' is null for any of the specified years, return False
        year_subset=group[group['Year']==year];

        if year_subset.empty or year_subset['PaidCircAllEds'].isnull().any() or (year_subset['PaidCircAllEds'] == 0).any():
            return False
    # If 'FreeCirculation' is not null for all specified years, return True
    return True


def check_condition_TotalCirc_Cross17_ID2(group):
    # Check the years
    years = [2017, 2018, 2019]
    for year in years:
        # If 'FreeCirculation' is null for any of the specified years, return False
        year_subset=group[group['Year']==year];

        if year_subset.empty or year_subset['TotalCircAllEds'].isnull().any() or (year_subset['TotalCircAllEds'] == 0).any():
            return False
    # If 'FreeCirculation' is not null for all specified years, return True
    return True


def check_condition_Rate_Cross17_ID2(group):
    # Check the year 2017
    year_2017 = group[group['Year'] == 2017]
    if year_2017.empty or year_2017['NatlLineRateAllEds'].isnull().any():
        return False

    # Check the years 2018 and 2019
    year_2018 = group[group['Year'] == 2018]
    year_2019 = group[group['Year'] == 2019]
    if ((year_2018.empty or year_2018['NatlLineRateAllEds'].isnull().all()) and
        (year_2019.empty or year_2019['NatlLineRateAllEds'].isnull().all())):
        return False

    # If 'NatlLineRateAllEds' is not null for year 2017, and for either 2018 or 2019, return True
    return True


# Apply the function to each group
FreeCirc_Cross17_ID2 = combined.groupby('ID2').apply(check_condition_FreeCirc_Cross17_ID2).reset_index(name='FreeCirc_Cross17_ID2')
FreeCirc_Cross17_ID2=FreeCirc_Cross17_ID2[FreeCirc_Cross17_ID2['FreeCirc_Cross17_ID2']]['ID2'].tolist()
combined['FreeCirc_Cross17_ID2']=0;
combined.loc[combined['ID2'].isin(FreeCirc_Cross17_ID2),'FreeCirc_Cross17_ID2']=1



# Apply the function to each group
PaidCirc_Cross17_ID2 = combined.groupby('ID2').apply(check_condition_PaidCirc_Cross17_ID2).reset_index(name='PaidCirc_Cross17_ID2')
PaidCirc_Cross17_ID2=PaidCirc_Cross17_ID2[PaidCirc_Cross17_ID2['PaidCirc_Cross17_ID2']]['ID2'].tolist()
combined['PaidCirc_Cross17_ID2']=0;
combined.loc[combined['ID2'].isin(PaidCirc_Cross17_ID2),'PaidCirc_Cross17_ID2']=1


Rate_Cross17_ID2 = combined.groupby('ID2').apply(check_condition_Rate_Cross17_ID2).reset_index(name='Rate_Cross17_ID2')
Rate_Cross17_ID2=Rate_Cross17_ID2[Rate_Cross17_ID2['Rate_Cross17_ID2']]['ID2'].tolist()
combined['Rate_Cross17_ID2']=0;
combined.loc[combined['ID2'].isin(Rate_Cross17_ID2),'Rate_Cross17_ID2']=1



TotalCirc_Cross17_ID2 = combined.groupby('ID2').apply(check_condition_TotalCirc_Cross17_ID2).reset_index(name='TotalCirc_Cross17_ID2')
TotalCirc_Cross17_ID2=TotalCirc_Cross17_ID2[TotalCirc_Cross17_ID2['TotalCirc_Cross17_ID2']]['ID2'].tolist()
combined['TotalCirc_Cross17_ID2']=0;
combined.loc[combined['ID2'].isin(TotalCirc_Cross17_ID2),'TotalCirc_Cross17_ID2']=1;




def condition2(df):
    post2017 = df[df['Year'] > 2017]
    return not post2017


OpPost2017 = combined.groupby('ID').apply(condition).reset_index(name='OpPost2017')
# Get IDs of newspapers to keep
OpPost2017 = OpPost2017[OpPost2017['OpPost2017']]['ID'].unique().tolist()
combined['OpPost2017']=0
# Filter the DataFrame
combined.loc[combined['ID'].isin(OpPost2017),'OpPost2017']=1

combined.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\combined.csv')

combined=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\combined.csv')

import ast

combined['CSDs'] = combined['CSDs'].apply(ast.literal_eval)

#### merge to get Census Division
CDSmapCD=pd.read_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\CDSmapCD.csv')


# Assuming df1 is your first dataframe and df2 is your second dataframe

# Convert df2 to a dictionary for faster lookups
CDSmapCD['CSDs']=CDSmapCD['CSDs'].str.strip()
CDSmapCD['CDs']=CDSmapCD['CDs'].str.strip()
map_dict = CDSmapCD.set_index('CSDs')['CDs'].to_dict()


with open(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\CDSmapCD.pkl', 'wb') as file:
    pickle.dump(map_dict, file)


#combined['CSDs'] = combined['CSDs'].apply(
#    lambda x: [element.strip(" '\"") for element in x.strip("[]").split(",")])

# Function to map a list of CSDs to a list of CDs
def map_csd_to_cd(csd_list):
    for csd in csd_list:
        print(csd)
    return [map_dict.get(csd) for csd in csd_list]


def check(csd_list):
    for csd in csd_list:
        print(csd)

def check_element(element_list, element_set):
    for element in element_list:
        if element in element_set:
            return 1
    return 0


# Create new CDs column
combined['CDs'] = combined['CSDs'].apply(map_csd_to_cd)


dfSwappedMarket=combined[(combined['Swap2017']==1) & (combined['Year'].isin([2013,2014,2015,2016,2017,2018,2019]))]
dfSwappedMarket.sort_values(by=['Newspaper','Year'],inplace=True)
dfSwappedMarket.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\checkSwapped.csv')

ListSwappedNewspaper = dfSwappedMarket.sort_values('MarketAndNewspaper')['MarketAndNewspaper'].unique().tolist()

with open(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\output.txt', 'w') as file:
    # Write each element of the list to the file
    for item in ListSwappedNewspaper:
        file.write(str(item) + '\n')


print('List Of Swapped Newspapers')
print(ListSwappedNewspaper)
print(len(ListSwappedNewspaper))
SwappedMarket=set(dfSwappedMarket.explode('CSDs')['CSDs'])
print('Number Of Affected CSDs')
print(len(SwappedMarket))

AllMarket=set(combined.explode('CSDs')['CSDs'])
print('Number Of All CSDs')
print(len(AllMarket))

with open(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\\affected_market.pkl', 'wb') as file:
    # Serialize and write the list to the file
    pickle.dump(SwappedMarket, file)


with open(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\\All_market.pkl', 'wb') as file:
    # Serialize and write the list to the file
    pickle.dump(AllMarket, file)




check=dfSwappedMarket[['MarketAndNewspaper','CSDs','Market','Year']]
check.sort_values(by=['MarketAndNewspaper','Year'],inplace=True)
check.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Check.csv')


combined['Swap2017Affected'] = combined['CSDs'].apply(lambda x: check_element(x, SwappedMarket))



num_unique_ids = combined['ID'].nunique()
print("Number of unique ID: ", num_unique_ids)
print("Number of Swapped Newspaper Editions Covered in the Data:",combined[combined['Swap2017']==1]['ID'].nunique())
print("Number of Newspaper Editions Affected in the Data:",combined[combined['Swap2017Affected']==1]['ID'].nunique())

num_unique_ids = combined['ID2'].nunique()
print("Number of unique ID2: ", num_unique_ids)
print("Number of Swapped Newspaper Covered in the Data:",combined[combined['Swap2017']==1]['ID2'].nunique())
print("Number of Newspaper Affected in the Data:",combined[combined['Swap2017Affected']==1]['ID2'].nunique())
combined.sort_values(by=['ID2','Year'])
combined.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Combined.csv')


























#MergeACSubCol=MergeAdvAndCirc[['ID','Year','FreeCirculation','TotalCirculation','PaidCirculation','Swap2017','Swap2017Affected','Owner','AdvCirc','NatlLineRate','SpotColour','ProcessColour','CSDs']]
MergeACSubCol=MergeAdvAndCirc
MergeACSubCol.sort_values(by=['ID','Year'],inplace=True)
#### study the Merged Dataset, Include All Merged Newspapers ##############
columns_to_exclude = ['ID','Year']  # List of columns to exclude
for id in unique_ids_1:
    id_data = MergeACSubCol[MergeACSubCol['ID'] == id]
    min_year, max_year = id_data['Year'].min(), id_data['Year'].max()
    for year in range(min_year + 1, max_year):
        if year not in id_data['Year'].values:
            new_row_data = {'ID': [id], 'Year': [year]}
            for column in MergeACSubCol.columns:
                if column not in columns_to_exclude:
                    new_row_data[column] = [np.nan]
            new_row = pd.DataFrame(new_row_data)
            MergeACSubCol = pd.concat([MergeACSubCol, new_row], ignore_index=True)


def interpolate_column(df, column):
    return df[column].interpolate(method='linear')
MergeACSubCol.sort_values(by=['ID','Year'],inplace=True)

for column in linear_interpolation_cols:
        MergeACSubCol[column] = MergeACSubCol.groupby('ID').apply(interpolate_column, column=column).reset_index(level=0, drop=True)
MergeACSubCol.sort_values(by=['ID','Year'],inplace=True)
MergeACSubCol[other_cols] = MergeACSubCol.groupby('ID')[other_cols].ffill()
MergeACSubCol.drop(columns=MergeACSubCol.columns[MergeACSubCol.columns.str.contains('Unnamed')], inplace=True)
MergeACSubCol.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\MergeACSubCol.csv', index=False)


IDwith2018or2019CircAvailible = MergeAdvAndCirc[MergeAdvAndCirc['Year'].isin([2018, 2019]) & (~MergeAdvAndCirc['TotalCirculation'].isna())]['ID'].unique()
Mdf4Affected=MergeAdvAndCirc[MergeAdvAndCirc['ID'].isin(IDwith2018or2019CircAvailible)]
Mdf4Affected.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\check.csv')
# Assuming your DataFrame is named 'Mdf4Affected' and contains a 'Newspaper' column
Mdf4Affected=Mdf4Affected[['ID','Year','FreeCirculation','TotalCirculation','PaidCirculation','Swap2017Affected','Owner','AdvCirc','NatlLineRate','SpotColour','ProcessColour','MarketAndNewspaper','AdvNewspaper','Newspaper','CSDs']]
Mdf4Affected = Mdf4Affected.sort_values(['ID', 'Year'])
unique_ids = Mdf4Affected['ID'].unique()


for id in unique_ids:
    id_data = Mdf4Affected[Mdf4Affected['ID'] == id]
    min_year, max_year = id_data['Year'].min(), id_data['Year'].max()
    for year in range(min_year+1, max_year):
        if year not in id_data['Year'].values:
            new_row = pd.DataFrame({'ID': [id], 'Year': [year], 'FreeCirculation': [np.nan], 'TotalCirculation': [np.nan], 'PaidCirculation': [np.nan],'Swap2017Affected': [np.nan],'Owner': [np.nan],'AdvCirc':[np.nan],'NatlLineRate':[np.nan],'SpotColour':[np.nan],'ProcessColour':[np.nan],'MarketAndNewspaper':[np.nan],'AdvNewspaper':[np.nan],'Newspaper':[np.nan],'CSDs':[np.nan]})
            new_row.reset_index(drop=True, inplace=True)
            Mdf4Affected = pd.concat([Mdf4Affected, new_row], ignore_index=True)

newspapers_to_drop = Mdf4Affected[Mdf4Affected['PaidCirculation'].notna() & (Mdf4Affected['PaidCirculation'] > 0)]['ID'].unique()
FreeNewspapers = Mdf4Affected[~Mdf4Affected['ID'].isin(newspapers_to_drop)]
FreeNewspapers.reset_index(drop=False, inplace=True)
FreeNewspapers.sort_values(by=['ID', 'Year'], inplace=True)
# Extract the year from the 'Year' column
FreeNewspapers['Year_only'] = FreeNewspapers['Year'].astype(int)





FreeNewspapers.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Mdf4AffectedBySwap2.csv')
def interpolate_column(df, column):
    return df[column].interpolate(method='linear')
for col in ['ID','Year','FreeCirculation','TotalCirculation','PaidCirculation','Swap2017Affected','Owner','AdvCirc','NatlLineRate','SpotColour','ProcessColour','CSDs']:
    FreeNewspapers[col] = FreeNewspapers.groupby('ID').apply(interpolate_column, column=col).reset_index(level=0, drop=True)

FreeNewspapers.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Mdf4AffectedBySwap.csv', index=False)


## Do a Box Plot for each year circulation:




# Filter the grouped data to include only newspapers with observations for all years from 2013 to 2019
# Count the number of newspapers with observations for all years from 2013 to 2019
#count_newspapers_with_2013_to_2019_observations = len(newspapers_with_2013_to_2019_observations)
#print("Number of newspapers with observations from 2013 to 2019:", count_newspapers_with_2013_to_2019_observations)
# Filter the dataframe for 2018 and 2019 and check for non-missing circulation data
#newspapers_with_data = Mdf4[Mdf4['Year'].isin([2018, 2019]) & (~Mdf4['TotalCirculation'].isna())]['MarketAndNewspaper'].nunique()
#print("Number of newspapers with circulation data for 2018 or 2019:", newspapers_with_data)


Mdf4.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\checkMdf4missingValues.csv')





Mdf4SortedByMarketAndNewspaper= Mdf4.sort_values(by=['MarketAndNewspaper', 'Year'], ascending=[True, True])
Mdf4SortedByMarketAndNewspaper=Mdf4SortedByMarketAndNewspaper[['EDITION','Year','Market','Newspaper','MarketAndNewspaper','TotalCirculation','Owner']]
Mdf4SortedByMarketAndNewspaper.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Mdf4SortedByMarketAndNewspaper.csv')
unique_Mdf4_MarketAndNewspaper=pd.DataFrame(Mdf4['MarketAndNewspaper'].unique(), columns=['MarketAndNewspaper'])
unique_Mdf4_MarketAndNewspaper.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\unique_Mdf4_MarketAndNewspaper.csv')




Mdf4.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\Mdf5.csv')

## Advertising Data





print(advs1['Issued'].unique())
unique_advs1_name=pd.DataFrame(advs1['Newspaper'].unique(), columns=['Newspaper'])
advs1_with_slash=pd.DataFrame(advs1.loc[advs1['Newspaper'].str.contains('/'),'Newspaper'].unique(), columns=['Newspaper'])
advs1_with_slash.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\advs1_with_slash.csv')
unique_advs1_name.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\unique_advs1_name.csv')

#unique_NewspapersMdf4.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\unique_NewspapersMdf4.csv')

unique_NewspapersAdvertising = Mdf4['Newspaper'].unique()
# Group the dataset by newspaper names and count the unique years for each newspaper
newspaper_years = Mdf4.groupby('Newspaper')['Year'].nunique()
# Filter the dataset to include only newspapers with only 2013 data
newspapers_2013_only2 = newspaper_years[newspaper_years == 1]
newspapers_2013_only2.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\newspapers_2013_only2.csv")
# Find newspaper names with forward slash
newspapers_with_slash = Mdf4[Mdf4['Newspaper'].str.contains('/')]
newspapers_with_slash=newspapers_with_slash[['Newspaper','Market','Owner']]
newspapers_with_slash.to_csv(fr'{current_drive}\IOnewspaper\openaipdf\CSD2016\newspapers_with_slash.csv")

