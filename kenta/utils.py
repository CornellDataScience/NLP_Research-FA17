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
