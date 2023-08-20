import pandas as pd
from fuzzywuzzy import fuzz


def match_dataframes(df1, df2, threshold):
    matches = []

    # Create dictionaries to store the best matches for each name
    best_match_dict1 = {}
    best_match_dict2 = {}

    # Create DataFrames to store unmatched values
    unmatched_df1 = df1.copy()
    unmatched_df2 = df2.copy()

    # Iterate over each name in the first dataframe
    for name1 in df1['Newspaper']:
        exact_match = None
        best_match1 = None
        best_similarity1 = 0

        # Iterate over each name in the second dataframe
        for name2 in df2['MarketAndNewspaper']:
            # Check for exact match
            if name1 == name2:
                exact_match = name2
                unmatched_df1 = unmatched_df1[unmatched_df1['Newspaper'] != name1]
                unmatched_df2 = unmatched_df2[unmatched_df2['MarketAndNewspaper'] != name2]
                break

            # Calculate the similarity between the two names
            similarity = fuzz.token_set_ratio(name1, name2)

            # Check if the similarity is above the threshold and better than the current best match for name1
            if similarity >= threshold and similarity > best_similarity1:
                # Check if the second name is not already matched with a better name1
                if name2 not in best_match_dict1 or similarity > best_match_dict1[name2][0]:
                    # Check if the first name is already matched with another name2
                    if best_match_dict1.get(name2) is None:
                        best_match1 = name2
                        best_similarity1 = similarity
                    else:
                        # Check if the similarity is better than the existing match
                        existing_similarity = best_match_dict1[name2][0]
                        if similarity > existing_similarity:
                            # Remove the existing match from best_match_dict2
                            existing_name1 = best_match_dict1[name2][1]
                            del best_match_dict2[existing_name1]

                            # Update the match for name1 in best_match_dict1
                            best_match1 = name2
                            best_similarity1 = similarity

        # Update the best match for name1 in the dictionary
        if exact_match:
            best_match_dict1[exact_match] = (100, name1)
            best_match_dict2[name1] = (100, exact_match)
        elif best_match1:
            best_match_dict1[best_match1] = (best_similarity1, name1)
            best_match_dict2[name1] = (best_similarity1, best_match1)

    # Combine the dictionaries to get the best matches
    matches = [(best_match_dict1[name2][1], name2) for name2 in best_match_dict1]

    # Create DataFrames for unmatched values

    second_elements1 = [value[1] for value in best_match_dict1.values()]
    second_elements2 = [value[1] for value in best_match_dict2.values()]

    unmatched_df1 = unmatched_df1[~unmatched_df1['Newspaper'].isin(second_elements1)]
    unmatched_df2 = unmatched_df2[~unmatched_df2['MarketAndNewspaper'].isin(second_elements2)]
    return matches, unmatched_df1, unmatched_df2




threshold = 0

df1=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\unique_advs1_name.csv')
df2=pd.read_csv(r'E:\IOnewspaper\openaipdf\CSD2016\unique_Mdf4_MarketAndNewspaper.csv')
ReturnValue=match_dataframes(df1, df2, threshold)
matches = ReturnValue[0]
unmatched_df1 = ReturnValue[1]
unmatched_df2 = ReturnValue[2]



matches=pd.DataFrame(matches, columns=["AdvNewspaper","MarketAndNewspaper"])
matches.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\FuzzyMatchAdvCirc.csv')
unmatched_df1.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\AdvUnmatched_df1.csv')
unmatched_df2.to_csv(r'E:\IOnewspaper\openaipdf\CSD2016\CircUnmatched_df2.csv')


