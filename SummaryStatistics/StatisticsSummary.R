library("readxl")
library("xlsx")
library("plyr")
library("dplyr")
library(magrittr)
library("Hmisc")
library("raster")
library("data.table")

df=read_excel("C:/Users/lenovo/Dropbox/IO project/PerfectFuzzyDataL/SufficentVariationOrNotEntryExit/MergedWithTotalFreeShareNoBundle/FreeContainBothNoBundleFreeNewspaperCompetitionMeasureMergedOneIssuePerWeekDemographic.xlsx")
count(df,"Lowner")
count(df,"Market")



df$Corporation=df$Lowner=="metroland media group ltd."|df$Lowner=="london publishing corporation"|df$Lowner=="postmedia network inc."|df$Lowner=="sun media corporation"
df$CorDummy=0
df$CorDummy[df$Corporation]=1
df$PercentageTotalShare=df$ControlledCirculation/df$TotalFreeCirculationByMarket

df$PercentageTotalShare=replace(df$PercentageTotalShare, is.nan(df$PercentageTotalShare), NA)

df$PercentageTotalShare[df$PercentageTotalShare>1] = 1
df$RateByCirculation=df$NatlLineRate/df$ControlledCirculation
df$RateByCirculation=replace(df$RateByCirculation,is.infinite(df$RateByCirculation),NA)
df$CompetitionFreeMeasure=df$CompetitorFreeCirculation/df$'Family characteristics Total - Census families in private households by family size - 100% data Character'




# Extend Demographic Characteristics 

df$PercentMover=df$'Mobility status - Place of residence 1 year ago Movers Character'/df$'Population and dwellings Population; 2016 Character'



aggregate(df$NatlLineRate, list(Corporation=df$Corporation,Date=df$Date), mean, na.rm=TRUE)
aggregate(df$ControlledCirculation,list(Corporation=df$Corporation,Date=df$Date),mean,na.rm=TRUE)
aggregate(df$NatlLineRate, list(Lowner=df$Lowner,Date=df$Date), mean, na.rm=TRUE)
aggregate(df$ControlledCirculation,list(Lowner=df$Lowner,Date=df$Date),mean,na.rm=TRUE)

##Mean CompeitionMeasure By Lowner ####
aggregate(df$CompetitorFreeCirculation,list(Lowner=df$Lowner,Date=df$Date),mean,na.rm=TRUE)
##Mean CompeitionMeasure By Corporation


aggregate(df$CompetitionFreeMeasure,list(Corporation=df$Corporation,Date=df$Date),mean,na.rm=TRUE)


x=dplyr::select(df,PercentMover,'Income of individuals in 2015 Average total income in 2015 among recipients ($) Character',CompetitionFreeMeasure,CorDummy,ControlledCirculation,NatlLineRate,TotalFreeCirculationByMarket,'Age characteristics Average age of the population Character','Labour force status Employment rate Character','Population and dwellings Population density per square kilometre Character','Population and dwellings Population; 2016 Character','Income of individuals in 2015 Average total income in 2015 among recipients ($) Character',PercentageTotalShare,RateByCirculation,
'Family characteristics Total - Census families in private households by family size - 100% data Character')

z=cor(x,use = "pairwise.complete.obs")

write.csv(z, "C:/Users/lenovo/Dropbox/IO project/SummaryStatistics/RegressionResult/CorMatrix.csv")

aggregate(df$NatlLineRate, list(Date=df$Date), mean, na.rm=TRUE)
aggregate(df$ControlledCirculation,list(Date=df$Date),mean,na.rm=TRUE)


##  Summarize Statistics for The Data Set ContainBothNoBundleFreeNewspaperCompetitionMeasureDemographic.xlsx
df1=read_excel("C:/Users/hongkong/Dropbox/IO project/PerfectFuzzyDataL/SufficentVariationOrNotEntryExit/MergedWithTotalFreeShareNoBundle/ContainBothNoBundleFreeNewspaperCompetitionMeasureDemographic.xlsx")
df1=setDT(df1)
df1[,print(.SD),by=.(Market,Date),.SDcols="Lowner"]

