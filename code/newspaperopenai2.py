import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import time


from geopy.geocoders import Nominatim
import requests
import geopandas as gpd
from shapely.geometry import Point


# Initialize geocoder
from geopy.geocoders import Nominatim



df=pd.read_csv("E:\IOnewspaper\IOnewspaper\Circulation\TheWholeCirculationNewspaperOwnerNameStandardized2023V.csv")

markets=df['Market']

MarketProv = {}


df['Address']=''

for i in range(len(df['Market'])):
    Prov=df['Prov'][i]
    market=df['Market'][i]
    addresses=[]
    for place in market.split("&"):
        place=place.replace("%20"," ")
        MarketProv[place] = Prov
        strAddress=place+" ,"+Prov+", Canada"
        addresses.append(strAddress)

    df['Address'][i]=addresses


df.to_csv("E:\IOnewspaper\IOnewspaper\Circulation\TheWholeCirculationNewspaperOwnerNameStandardized2023V2.csv")





#def get_location_by_name(name):
#    geolocator = Nominatim(user_agent="geoapiExercises")
#    location = geolocator.geocode(name)
#    return location.latitude, location.longitude


#LocationGPS={}
#for key, value in MarketProv.items():
#    Location=key+" ,"+value+", Canada"
#    try:
#        GPS = get_location_by_name(Location)
#        print(GPS)
#        LocationGPS[Location]=GPS
#   except:
#        print("problem")
#        print(Location)
#        LocationGPS[Location] = (1000,1000)
#        pass
#    time.sleep(1)

import pickle

#with open('LocationGPS.pkl', 'wb') as file:
#    pickle.dump(LocationGPS, file)

# Read the dictionary from the file
with open('LocationGPS.pkl', 'rb') as file:
    LocationGPS= pickle.load(file)



import geopandas as gpd
#from shapely.geometry import Point

# Load the shapefile
# Note: You need to replace 'path_to_shapefile' with the actual path to the shapefile


#CSDShape= gpd.read_file(r'E:\IOnewspaper\openaipdf\CSD2016\lcsd000b16a_e.shp')

#CSDShape = CSDShape.to_crs("EPSG:4326")

#CSDShape.to_pickle('CSDShape.pkl')

# To load
#CSDShape = pd.read_pickle(r'E:\IOnewspaper\openaipdf\CSD2016\CSDShape.pkl')

#columns1=['Address','Longitude','Latitude','CSDUID', 'CSDNAME', 'CSDTYPE', 'PRUID', 'PRNAME', 'CDUID', 'CDNAME',
#       'CDTYPE', 'CCSUID', 'CCSNAME', 'ERUID', 'ERNAME', 'SACCODE', 'SACTYPE',
#       'CMAUID', 'CMAPUID', 'CMANAME', 'CMATYPE']

#CSDdf=pd.DataFrame(columns=columns1)

#for Key,Value in LocationGPS.items():
#    point = Point(Value[1], Value[0])
#    print(point)
#    subdivision = CSDShape[CSDShape.geometry.contains(point)]
#    print(subdivision)
#    subdivision2=subdivision[['CSDUID', 'CSDNAME', 'CSDTYPE', 'PRUID', 'PRNAME', 'CDUID', 'CDNAME',
#       'CDTYPE', 'CCSUID', 'CCSNAME', 'ERUID', 'ERNAME', 'SACCODE', 'SACTYPE',
#       'CMAUID', 'CMAPUID', 'CMANAME', 'CMATYPE']]
#    subdivision2['Address']=Key
#    subdivision2['Longitude'] = Value[1]
#    subdivision2['Latitude'] = Value[0]
#    CSDdf=pd.concat([CSDdf,subdivision2],axis=0)

#from unidecode import unidecode


#CSDdf['CSDNAME']=CSDdf['CSDNAME'].apply(lambda x: unidecode(str(x)))

#CSDdf.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf.csv')


CSDdf=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CSDdf.csv',index_col=0)


CSDdf['CSDNAME'] = CSDdf['CSDNAME'].str.strip()
CSDdf['PRNAME'] = CSDdf['PRNAME'].str.strip()
## deal with Chass Data




Chass2Header = 'E:\IOnewspaper\openaipdf\CSD2016\Chass2Header.txt'

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



dfChass2 =pd.read_csv('E:\IOnewspaper\openaipdf\CSD2016\Chass2.csv')

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

CSDdf['PRNAME']=CSDdf['PRNAME'].map(province_mapping)

dfChass2['CSD name'] = dfChass2['CSD name'].str.strip()


CSDdf2=pd.merge(CSDdf,dfChass2,left_on=['CSDNAME','PRNAME'],right_on=['CSD name','Province name'], how='left')


CSDdf2.to_csv('E:\IOnewspaper\openaipdf\CSD2016\CSDdf2.csv')


circulation=pd.read_csv(r'E:\IOnewspaper\IOnewspaper\Circulation\TheWholeCirculationNewspaperOwnerNameStandardized2023V2.csv',index_col=0)

import ast

#CSDdf3 = pd.DataFrame()



#for index, row in circulation.iterrows():
#    print(index)
#    ele1 = row['Address']
#    # Assuming the list is stored as a string, we convert it to a list
#    subdf2 = CSDdf2[CSDdf2['Address'].isin(ast.literal_eval(ele1))]
#    population_index = subdf2.columns.get_loc("Population and dwelling counts / Population, 2016")
#    subdf3 = subdf2.iloc[:, population_index:]
#    mean_values = subdf3.mean(skipna=True)
#    result_df = pd.DataFrame([mean_values], columns=mean_values.index)
#    row_df = pd.DataFrame([row], columns=row.index)
#    row_df.reset_index(drop=True,inplace=True)
#    result_df.reset_index(drop=True,inplace=True)
#    concatenated_row = row_df.join(result_df)
#    CSDdf3 = pd.concat([CSDdf3, concatenated_row], axis=0, ignore_index=True)

#CSDdf3.to_csv("CSDdf3.csv")

CSDdf3=pd.read_csv("E:\IOnewspaper\openaipdf\CSD2016\CSDdf3.csv",index_col=0)

CSDdf3['GPS']=CSDdf3['Address'].apply(lambda x: [LocationGPS[i] for i in ast.literal_eval(x)])

ACSDDictionary = {}

AddressCSD=CSDdf2[['Address','CSDNAME']]

# Iterate over the DataFrame rows
for index, row in AddressCSD.iterrows():
    key = row['Address']
    value = row['CSDNAME']
    ACSDDictionary[key] = value

print(ACSDDictionary)
CSDdf3['CSDs']=CSDdf3['Address'].apply(lambda x: [ACSDDictionary[i] for i in ast.literal_eval(x) if i in ACSDDictionary])
CSDdf4=CSDdf3
CSDdf4.to_csv(r"E:\IOnewspaper\openaipdf\CSD2016\CSDdf4.csv")






import requests

# Define the parameters
latitude = 45.5017
longitude = -73.5673
year = 2021

# Use the Statistics Canada API to get the census subdivision code
url = f"https://geocoder.api.statcan.gc.ca/geocodews/rs/geocode?latt={latitude}&longt={longitude}&geoMT=Y&geoSearch=FS"
response = requests.get(url)
census_subdivision_code = response.json()["GeoResult"]["censusDivision"]["censusSubdivision"]["@csdCode"]

# Use the Statistics Canada API to get the population for the specified year
url = f"https://www12.statcan.gc.ca/rest/census-recensement/CPR2016.json?lang=E&dguid={census_subdivision_code}&topic=13&notes=0&stat=0"
response = requests.get(url)
data = response.json()

# Find the population for the specified year
for population in data["DATA"]:
    if population[1] == year:
        print(f"The population in {population[4]} in {year} is {population[10]}.")
        break




import openpyxl
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
root_dir = "/content/drive/MyDrive/openai"

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os

os.chdir('/content/drive/MyDrive/openai')

agent = create_csv_agent(OpenAI(temperature=3), 'TheWholeCirculationNewspaperOwnerNameStandardized2023V.csv', verbose=True)

agent.run("The market column contains some locations that located in a Canadian province indicated in the Prov column,where %20 means Space of the string and & is seperate that indicate a seperate location, for each market, find the longitide and latitude, and calcualte the average longtitude and latitude of the places in each market, add two columns longtitude and latitude to the data set and store the longitude in the column longitude and latitude in column latitude, save to df.csv in the same folder ")