import numpy as np
import pandas as pd
import re



df2013=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2013.xlsx')
df2014=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2014.xlsx')
df2015=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2015.xlsx')
df2016=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2016.xlsx')
df2017=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2017.xlsx')
df2018=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2018.xlsx')
df2019=pd.read_excel(r'C:\Users\hongkong\Dropbox\IO project\Circulation\circulation2019.xlsx')
df2013['Date']=2013
df2014['Date']=2014
df2015['Date']=2015
df2016['Date']=2016
df2017['Date']=2017
df2018['Date']=2018
df2019['Date']=2019


Frames=[df2013,df2014,df2015,df2016,df2017,df2018,df2019]
Frames=pd.concat(Frames)

df=Frames





#df=Frames[Frames['PaidCirculation']/Frames['TotalCirculation']<0.1]
#df=df.groupby(['Market'])[['ControlledCirculation']].agg('sum')
#df=df.rename(columns={'ControlledCirculation':'TotalControlledCirculation'})
#df=df.reset_index()



def StandardizeMarket(x):
    c=x.str.strip()
    c=c.str.lower()
    c=c.str.replace(' and area','')
    c=c.str.replace('county','')
    c=c.str.replace('region','')
    c=c.str.strip()
    c=c.str.replace(',','/')
    c=c.str.replace(' ','%20')
    c=c.str.strip()
    return c

df['Market']=StandardizeMarket(df['Market'])



# Standardize Market Names

df['Market']=df['Market'].str.strip()
df['Market']=df['Market'].str.lower()
df['Market']=df['Market'].str.replace('bradford/west gwillimbury','bradford west gwillimbury')
df['Market']=df['Market'].str.replace(' and area','')
df['Market']=df['Market'].str.replace('elmira-woolwich','elmira/woolwich')
df['Market']=df['Market'].str.replace('fergus-elora','fergus/elora')
df['Market']=df['Market'].str.replace('leamington-wheatley','leamington/wheatley')
df['Market']=df['Market'].str.replace('strathroy-middlesex','strathroy/middlesex')
df['Market']=df['Market'].str.replace('niagara-on-the-lake','niagara%20on%20the%20lake')


# Standardize Market Name for Total Free Market Share
df['Market']=StandardizeMarket(df['Market'])



df['Market']=df['Market'].str.replace('new%20tecumseh','new%20tecumseth')
df['Market']=df['Market'].str.replace('grimsby%20lincoln','grimsby/lincoln')
df['Market']=df['Market'].str.replace('/','&')












#First Step in Standardize Newspaper Name:


df['Newspaper']=df['Newspaper'].str.strip()
df['Newspaper']=df['Newspaper'].str.replace('(','')
df['Newspaper']=df['Newspaper'].str.replace(')','')
df['Newspaper']=df['Newspaper'].str.replace('-','')
df['Newspaper']=df['Newspaper'].str.replace('\'','')
df['Newspaper']=df['Newspaper'].str.replace('’','')
df['Newspaper']=df['Newspaper'].str.strip()
df['Newspaper']=df['Newspaper'].str.lower()
df['Newspaper'].unique()











#Second Step In StandarDize Newspaper Name:


df['Newspaper']=df['Newspaper'].str.replace('actionl','action l')
df['Newspaper']=df['Newspaper'].str.replace('arnprior chronicle','arnprior chronicle guide')
df['Newspaper']=df['Newspaper'].str.replace('arnprior chronicleguide','arnprior chronicle guide')
df['Newspaper']=df['Newspaper'].str.replace('arprior chronicleguide','arnprior chronicle guide')
df['Newspaper']=df['Newspaper'].str.replace('arprior chronicleguide emc','arnprior chronicle guide')

df['Newspaper']=df['Newspaper'].str.replace('arnprior chronicle guideguide','arnprior chronicle guide')
df['Newspaper']=df['Newspaper'].str.replace('arnprior chronicle guide emc','arnprior chronicle guide')



df['Newspaper']=df['Newspaper'].str.replace('belleville emc','belleville news')
df['Newspaper']=df['Newspaper'].str.replace('Bradford West GwillimburyTopic','Bradford West Gwillimbury Topic')
df['Newspaper']=df['Newspaper'].str.replace('brighton/warkworth/campbellford independent','brighton Independent')
df['Newspaper']=df['Newspaper'].str.replace('carleton place/almonte emc','carleton placealmonte canadian gazette')
df['Newspaper']=df['Newspaper'].str.replace('erabanner','era banner')
df['Newspaper']=df['Newspaper'].str.replace('frontenac emc','frontenac gazette')




df['Newspaper']=df['Newspaper'].str.replace('goût de vivre le','gout de vivre le')
df['Newspaper']=df['Newspaper'].str.replace('guelph tribune','guelph mercury tribune')
df['Newspaper']=df['Newspaper'].str.replace('kanata kourierstandard emc','kanata kourierstandard')
df['Newspaper']=df['Newspaper'].str.replace('kemptville emc','kemptville advance')
df['Newspaper']=df['Newspaper'].str.replace('kingston emc','kingston heritage')
df['Newspaper']=df['Newspaper'].str.replace('kitchener citizen, east editi','kitchener citizen, east edition')



df['Newspaper']=df['Newspaper'].str.replace('kitchener citizen, west editi','kitchener citizen, west edition')
df['Newspaper']=df['Newspaper'].str.replace('lake erie beac','lake erie beacon')
df['Newspaper']=df['Newspaper'].str.replace('le régional','le regional')
df['Newspaper']=df['Newspaper'].str.replace('manotick manotick emc','manotick news')
df['Newspaper']=df['Newspaper'].str.replace('marathon mercury','mercury')
df['Newspaper']=df['Newspaper'].str.replace('nepean / barrhaven emc','nepean / barrhaven news')



df['Newspaper']=df['Newspaper'].str.replace('nipigon/red rock gazette','nipigonred rock gazette')
df['Newspaper']=df['Newspaper'].str.replace('orleans emc','orleans news')
df['Newspaper']=df['Newspaper'].str.replace('ottawa east emc','ottawa east news')
df['Newspaper']=df['Newspaper'].str.replace('ottawa south emc','ottawa south news')
df['Newspaper']=df['Newspaper'].str.replace('ottawa west emc','ottawa west news')
df['Newspaper']=df['Newspaper'].str.replace('perth emc','perth courier')


df['Newspaper']=df['Newspaper'].str.replace('quinte west emc','quinte west news')
df['Newspaper']=df['Newspaper'].str.replace('renfrew mercury emc','renfrew mercury')
df['Newspaper']=df['Newspaper'].str.replace('sarnia/lambtthis week','sarnia/lambton this week')
df['Newspaper']=df['Newspaper'].str.replace('sault ste. marie this week','sault this week')
df['Newspaper']=df['Newspaper'].str.replace('shelburne free press','shelburne free press & economist')
df['Newspaper']=df['Newspaper'].str.replace('shoreline beac','shoreline beacon')


df['Newspaper']=df['Newspaper'].str.replace('smiths falls emc','smiths falls record news')
df['Newspaper']=df['Newspaper'].str.replace('snapd','snap')
df['Newspaper']=df['Newspaper'].str.replace('stittsville news emc','stittsville news')
df['Newspaper']=df['Newspaper'].str.replace('sun/tribune','suntribune')
df['Newspaper']=df['Newspaper'].str.replace('terrace bay/schreiber news','terrace bayschreiber news')
df['Newspaper']=df['Newspaper'].str.replace('the stayner sun','the sun')



df['Newspaper']=df['Newspaper'].str.replace('tillsonburg independent/news','tillsonburg independent')
df['Newspaper']=df['Newspaper'].str.replace('west carletreview emc','west carleton review')




########################################## Standardize Owners Names for Circulation Data #################################





ReOwner=re.compile('Metroland.*')
SunMedia=re.compile('Sun Media.*')

df['Owner']=df['Owner'].str.replace(ReOwner,'Metroland Media Group Ltd.')
df['Owner']=df['Owner'].str.replace('Snap Newspaper Group Inc.','snapd Inc.')
df['Owner']=df['Owner'].str.replace(SunMedia,'Sun Media Corporation')
df['Owner']=df['Owner'].str.replace('Quebecor Media','Sun Media Corporation')
df['Owner']=df['Owner'].str.replace('London Publishing/Claridge Newspaper','London Publishing Corporation')
df['Owner']=df['Owner'].str.replace('TC Media','TC.Transcontinental')
df['Owner']=df['Owner'].str.replace('La Compagnie d\'édition André Paquette Inc','Compagnie d\'Edition André Paquette')
df['Owner']=df['Owner'].str.replace('La Compagnie d\'édition André Paquette Inc','Compagnie d\'Edition André Paquette')
df['Owner']=df['Owner'].str.replace('Journal Le Nord Inc.','Le Nord Inc.')
df['Owner']=df['Owner'].str.replace('Community Bulletin Newspaper Group, Inc.','Community Bulletin Newspaper Group')
df['Owner']=df['Owner'].str.replace('Community Bulletin Newspaper Group, Inc.','Community Bulletin Newspaper Group')
df['Owner']=df['Owner'].str.replace('Etcetera Publications Inc.','Etcetera Publications')
df['Owner']=df['Owner'].str.replace('é','e')

df['Owner']=df['Owner'].str.strip()

df=df.reset_index(drop=True)



 # write owenr name into lower case
df['Owner']=df['Owner'].str.lower()


df=df[['Date','Market','Owner']]

################## Drop Null Value #####################################
df=df[(~df['Market'].isnull())&(~df['Owner'].isnull())]

df2013=df[df['Date']==2013]
df2013=df2013.sort_values(by=['Market','Owner'])
df2013.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2013.xlsx')

df2014=df[df['Date']==2014]
df2014=df2014.sort_values(by=['Market','Owner'])
df2014.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2014.xlsx')

df2015=df[df['Date']==2015]
df2015=df2015.sort_values(by=['Market','Owner'])
df2015.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2015.xlsx')

df2016=df[df['Date']==2016]
df2016=df2016.sort_values(by=['Market','Owner'])
df2016.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2016.xlsx')

df2017=df[df['Date']==2017]
df2017=df2017.sort_values(by=['Market','Owner'])
df2017.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2017.xlsx')

df2018=df[df['Date']==2018]
df2018=df2018.sort_values(by=['Market','Owner'])
df2018.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2018.xlsx')


df2019=df[df['Date']==2019]
df2019=df2019.sort_values(by=['Market','Owner'])
df2019.to_excel(r'C:\Users\hongkong\Dropbox\IO project\SummaryStatistics\ListOwnerByMarket\MarketOnwerList2019.xlsx')



market2013=df2013['Market'].unique()
MarketDiffOwner2013=df2013.groupby('Market')['Owner'].nunique()
MarketDiffOwner2013=pd.DataFrame({'Market':MarketDiffOwner2013.index, 'N':MarketDiffOwner2013.values})
MarketDiffOwner2013=MarketDiffOwner2013.groupby('N')['Market'].nunique()
MarketDiffOwner2013=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2013.index, 'NumberOfMarket':MarketDiffOwner2013.values})
df2013['#CompeitionCorporation']=0


for marketname in market2013:
    interested=df2013[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2013.at[df2013['Market']==marketname,'#CompeitionCorporation']=n
            






print("The Market Onwer List For 2013\n")
print(MarketDiffOwner2013)
print("The Total Number of Market is\n")
print(len(market2013))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2013[df2013['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2013))
print("Number of Companies\n")
print(len(df2013['Owner'].unique()))






market2014=df2014['Market'].unique()
MarketDiffOwner2014=df2014.groupby('Market')['Owner'].nunique()
MarketDiffOwner2014=pd.DataFrame({'Market':MarketDiffOwner2014.index, 'N':MarketDiffOwner2014.values})
MarketDiffOwner2014=MarketDiffOwner2014.groupby('N')['Market'].nunique()
MarketDiffOwner2014=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2014.index, 'NumberOfMarket':MarketDiffOwner2014.values})

df2014['#CompeitionCorporation']=0

for marketname in market2014:
    interested=df2014[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2014.at[df2014['Market']==marketname,'#CompeitionCorporation']=n
            







print("The Market Onwer List For 2014\n")
print(MarketDiffOwner2014)
print("The Total Number of Market is\n")
print(len(market2014))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2014[df2014['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2014))
print("Number of Companies\n")
print(len(df2014['Owner'].unique()))




                








market2015=df2015['Market'].unique()

MarketDiffOwner2015=df2015.groupby('Market')['Owner'].nunique()
MarketDiffOwner2015=pd.DataFrame({'Market':MarketDiffOwner2015.index, 'N':MarketDiffOwner2015.values})
MarketDiffOwner2015=MarketDiffOwner2015.groupby('N')['Market'].nunique()
MarketDiffOwner2015=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2015.index, 'NumberOfMarket':MarketDiffOwner2015.values})
df2015['#CompeitionCorporation']=0

for marketname in market2015:
    interested=df2015[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2015.at[df2015['Market']==marketname,'#CompeitionCorporation']=n
            



print("The Market Onwer List For 2015\n")
print(MarketDiffOwner2015)
print("The Total Number of Market is\n")
print(len(market2015))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2015[df2015['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2015))
print("Number of Companies\n")
print(len(df2015['Owner'].unique()))




                






market2016=df2016['Market'].unique()
MarketDiffOwner2016=df2016.groupby('Market')['Owner'].nunique()
MarketDiffOwner2016=pd.DataFrame({'Market':MarketDiffOwner2016.index, 'N':MarketDiffOwner2016.values})
MarketDiffOwner2016=MarketDiffOwner2016.groupby('N')['Market'].nunique()
MarketDiffOwner2016=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2016.index, 'NumberOfMarket':MarketDiffOwner2016.values})
df2016['#CompeitionCorporation']=0

for marketname in market2016:
    interested=df2016[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2016.at[df2016['Market']==marketname,'#CompeitionCorporation']=n
            




print("The Market Onwer List For 2016\n")
print(MarketDiffOwner2016)
print("The Total Number of Market is\n")
print(len(market2016))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2016[df2016['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2016))
print("Number of Companies\n")
print(len(df2016['Owner'].unique()))








market2017=df2017['Market'].unique()
MarketDiffOwner2017=df2017.groupby('Market')['Owner'].nunique()
MarketDiffOwner2017=pd.DataFrame({'Market':MarketDiffOwner2017.index, 'N':MarketDiffOwner2017.values})
MarketDiffOwner2017=MarketDiffOwner2017.groupby('N')['Market'].nunique()
MarketDiffOwner2017=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2017.index, 'NumberOfMarket':MarketDiffOwner2017.values})
df2017['#CompeitionCorporation']=0

for marketname in market2017:
    interested=df2017[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2017.at[df2017['Market']==marketname,'#CompeitionCorporation']=n
            

print("The Market Onwer List For 2017\n")
print(MarketDiffOwner2017)
print("The Total Number of Market is\n")
print(len(market2017))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2017[df2017['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2017))
print("Number of Companies\n")
print(len(df2017['Owner'].unique()))








market2018=df2018['Market'].unique()
MarketDiffOwner2018=df2018.groupby('Market')['Owner'].nunique()
MarketDiffOwner2018=pd.DataFrame({'Market':MarketDiffOwner2018.index, 'N':MarketDiffOwner2018.values})
MarketDiffOwner2018=MarketDiffOwner2018.groupby('N')['Market'].nunique()
MarketDiffOwner2018=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2018.index, 'NumberOfMarket':MarketDiffOwner2018.values})
df2018['#CompeitionCorporation']=0

for marketname in market2018:
    interested=df2018[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2018.at[df2018['Market']==marketname,'#CompeitionCorporation']=n
            



print("The Market Onwer List For 2018\n")
print(MarketDiffOwner2018)
print("The Total Number of Market is\n")
print(len(market2018))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2018[df2018['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2018))
print("Number of Companies\n")
print(len(df2018['Owner'].unique()))










market2019=df2019['Market'].unique()
MarketDiffOwner2019=df2019.groupby('Market')['Owner'].nunique()
MarketDiffOwner2019=pd.DataFrame({'Market':MarketDiffOwner2019.index, 'N':MarketDiffOwner2019.values})
MarketDiffOwner2019=MarketDiffOwner2019.groupby('N')['Market'].nunique()
MarketDiffOwner2019=pd.DataFrame({'NumberOfOwnerForEachMarket':MarketDiffOwner2019.index, 'NumberOfMarket':MarketDiffOwner2019.values})
df2019['#CompeitionCorporation']=0

for marketname in market2019:
    interested=df2019[df['Market']==marketname]
    InterestedMarketOwner=interested['Owner'].unique()
    n=0
    for InterestedOwnerName in InterestedMarketOwner:
        if((InterestedOwnerName=='metroland media group ltd.')|(InterestedOwnerName=='sun media corporation')|(InterestedOwnerName=='postmedia network inc.')):
            n=n+1;
            df2019.at[df2019['Market']==marketname,'#CompeitionCorporation']=n
            

print("The Market Onwer List For 2019\n")
print(MarketDiffOwner2019)
print("The Total Number of Market is\n")
print(len(market2019))
print("\n")
print("The Total Number of CompeitionCorporation more than 2\n")
print(len(df2019[df2019['#CompeitionCorporation']>1]))
print("Number of Newspapers\n")
print(len(df2019))
print("Number of Companies\n")
print(len(df2019['Owner'].unique()))




























