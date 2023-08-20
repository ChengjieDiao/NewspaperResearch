import requests

# Set the API endpoint and parameters
endpoint_url = 'https://www.statcan.gc.ca/eng/wds/api/data/'
census_year = '2016'
csd_guid = '3539036'
topic_id = '2'


# Construct the API request URL
api_url = f'{endpoint_url}{census_year}/25?lang=E&dguid={csd_guid}&topic={topic_id}'

# Make the API request
headers = {'Content-Type': 'application/json'}
response = requests.get(api_url, headers=headers)

# Retrieve the response data
data = response.json()
