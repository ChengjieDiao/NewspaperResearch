import pandas as pd
import numpy as np


Chass2Header = 'E:\IOnewspaper\openaipdf\CSDRMapperData\CSD16YFVariableHeader.txt'

data_dict = {}
with open(Chass2Header, 'r') as file:
    # Skip the first line
    next(file)
    # Read the second line and split by " - "
    x=file.readline()
    while ' - ' in x:
        values= x.strip().split(" - ")
        column_name =  values[0]
        column_description = values[1:]
        data_dict[column_name] = column_description
        # Read the next line
        x=file.readline()

for column_name, column_description in data_dict.items():
    print(column_description)
    data_dict[column_name] = " / ".join(column_description)



dfChass2 =pd.read_csv('E:\IOnewspaper\openaipdf\CSDRMapperData\CSD16YFVariable.csv', encoding='ISO-8859-1')

dfChass2.rename(columns=data_dict, inplace=True)
dfChass2['Province name']=dfChass2['Province name'].str.strip()

province_mapping = {
    'Newfoundland and Labrador': 'NL',
    'Prince Edward Island': 'PE',
    'Nova Scotia': 'NS',
    'New Brunswick': 'NB',
    'Quebec': 'QC',
    'Ontario': 'ON',
    'Manitoba': 'MB',
    'Saskatchewan': 'SK',
    'Alberta': 'AB',
    'British Columbia': 'BC',
    'Yukon': 'YT',
    'Northwest Territories': 'NT',
    'Nunavut': 'NU'
}

# Assuming your DataFrame is named 'df' and the column with province names is named 'Province'
dfChass2['Province name'] = dfChass2['Province name'].map(province_mapping)

dfChass2.to_csv(r'E:\IOnewspaper\openaipdf\CSDRMapperData\CSD16YFVariable-2.csv')