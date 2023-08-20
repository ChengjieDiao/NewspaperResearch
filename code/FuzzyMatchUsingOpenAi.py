import pandas as pd
import openai

# Set up your OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

def calculate_similarity(text1, text2):
    # Use OpenAI's text similarity API to calculate the similarity between two texts
    response = openai.ComparisonCompletion.create(
        model="text-davinci-003",
        texts=[text1, text2],
        options={"stop": ["\n"]},
    )

    similarity = response.choices[0].similarity

    return similarity

def match_dataframes(df1, df2, threshold):
    matches = []

    # Iterate over each name in the first dataframe
    for name1 in df1['Newspaper Name']:
        best_match = None
        best_similarity = 0

        # Iterate over each name in the second dataframe
        for name2 in df2['Newspaper Name']:
            # Calculate the similarity between the two names using OpenAI's text similarity model
            similarity = calculate_similarity(name1, name2)

            # Check if the similarity is above the threshold and better than the current best match
            if similarity >= threshold and similarity > best_similarity:
                # Check if the second name is not already matched with another name
                if name2 not in [match[1] for match in matches]:
                    best_match = name2
                    best_similarity = similarity

        # Add the best match to the matches list if it exists
        if best_match:
            matches.append((name1, best_match))

    return matches

# Set the threshold for similarity
threshold = 0.7

# Call the match_dataframes function
matches = match_dataframes(df1, df2, threshold)

# Print the matches
for match in matches:
    print(f"{match[0]} -> {match[1]}")



## Fuzz Match



import pandas as pd
from fuzzywuzzy import fuzz

def match_dataframes(df1, df2, threshold):
    matches = []

    # Iterate over each name in the first dataframe
    for name1 in df1['Newspaper Name']:
        best_match = None
        best_similarity = 0

        # Iterate over each name in the second dataframe
        for name2 in df2['Newspaper Name']:
            # Calculate the similarity between the two names
            similarity = fuzz.token_set_ratio(name1, name2)

            # Check if the similarity is above the threshold and better than the current best match
            if similarity >= threshold and similarity > best_similarity:
                # Check if the second name is not already matched with another name
                if name2 not in [match[1] for match in matches]:
                    best_match = name2
                    best_similarity = similarity

        # Add the best match to the matches list if it exists
        if best_match:
            matches.append((name1, best_match))

    return matches

# Set the threshold for similarity
threshold = 80

# Call the match_dataframes function
matches = match_dataframes(df1, df2, threshold)

# Print the matches
for match in matches:
    print(f"{match[0]} -> {match[1]}")

