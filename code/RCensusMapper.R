# Load the library
library(cancensus)
library

# Specify the CensusMapper API key
options(cancensus.api_key = "CensusMapper_f6d390223f326bcd61bb4b35d0405508")

# Specify the dataset, regions, and variables you want to retrieve
dataset <- "CA16" # 2016 Canadian Census
regions <- list(CSD = "all") # all census subdivisions
variables <- list(TOTAL = "v_CA16_408") # Total - Age groups and average age of the population

# Retrieve the data
census_data <- get_census(dataset='CA16',
                          vectors=c("v_CA16_401","v_CA16_379","v_CA16_2207","v_CA16_5078"),regions=list(PR="Ontario"),
                          level='CSD', use_cache = FALSE, geo_format = NA, quiet = TRUE)

variables <- list_census_vectors("CA16")
print(variables, n = Inf)


write.csv(variables, file = "E:\\IOnewspaper\\openaipdf\\CSDRMapperData\\census_variables.csv", row.names = FALSE)

write.csv(variables, file = "E:\\IOnewspaper\\openaipdf\\CSDRMapperData\\CSDDemog16.csv", row.names = FALSE)