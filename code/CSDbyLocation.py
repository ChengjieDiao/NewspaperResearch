import geopandas as gpd
from shapely.geometry import Point

# Load the shapefile
# Note: You need to replace 'path_to_shapefile' with the actual path to the shapefile
canada = gpd.read_file('path_to_shapefile.shp')

# Create a geographic point based on longitude and latitude
# Note: Replace 'longitude' and 'latitude' with actual coordinates
point = Point(longitude, latitude)

# Find the census subdivision that contains the point
subdivision = canada[canada.geometry.contains(point)]

print(subdivision)


## get city by longitude and location


def get_location_by_name(name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(name)
    return location.latitude, location.longitude

print(get_location_by_name("bradford west gwillimbury ,ON, Canada"))




from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

lat=44.1143279
long=-79.5647069

def get_location_by_coordinates(lat, long):
    location = geolocator.reverse([lat, long])
    return location.raw

# Replace 'latitude' and 'longitude' with your actual data
location = get_location_by_coordinates(lat, long)
print(location)


city = location['address'].get('city', '')
print(city)



## get the municipality population


import requests
import pandas as pd


def get_census_data(year):
    # Define the base URL for the API
    base_url = "https://www12.statcan.gc.ca/rest/census-recensement/CPR2016.json"

    # Define the parameters for the API request
    # Note: The 'topic' parameter may need to be adjusted depending on the specific data you want
    params = {
        "lang": "E",
        "dguid": "2016A000011124",  # This is an example geographical code (Canada as a whole)
        "topic": "1",  # This is the topic code for population and dwelling counts
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    # The response is in JSONP format, so we need to remove the leading 'jsonp(' and trailing ')'
    data = response.text[6:-1]

    # Convert the data to JSON
    data = json.loads(data)

    # The relevant data is in the 'DATA' field, and we can convert this to a pandas DataFrame
    df = pd.DataFrame(data['DATA'], columns=data['COLUMNS'])

    return df


# Get the 2016 Census data
census_data = get_census_data(2016)

print(census_data)
