import pandas as pd
import json

def load_json_to_df(datapass):
    '''
    Load the json file and parse the file to pandas dataframe format

    Input:
        datapass(str) : directory to the json file
    Output:
        df(dataframe) : pandas dataframe object
    '''

    data = []
    with open(datapass) as data_file:
        for f in data_file:
            data.append(json.loads(f))
    df = pd.DataFrame(data)
    return df

def category_count(df):
    '''
    return a dictionary that maps category to the unique business count

    Input:
        df(dataframe) : business dataframe with the column name 'categories'

    Output:
        categories(list) : sorted list of tuple ('category', business)
    '''
    categories = {}
    for C in df.categories:
        for c in C:
            if c in categories:
                categories[c] += 1
            else:
                categories[c] = 1

    categories = sorted(categories.items(), key=lambda x: x[1], reverse = True)
    return categories


def business_id_retrieval(cat, business):
    '''
    Input:
        cat(str) : category
        business(dataframe) : the business data
    Output:
        id_list(set) : business ids of a particular category
    '''
    id_list = set()
    idx = 0
    for row in business.values:
        categories = row[3]
        if cat in categories:
            id_list.add(row[2])
    return id_list
