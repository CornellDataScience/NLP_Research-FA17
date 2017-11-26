import nltk
import pandas as pd
import numpy as np
import json


print("Lalalalala")

data = json.load(open("/home/cai29/Cornell/CDS/yelp/assets/dataset/review.json"))
d = pd.DataFrame(data, columns=['user_id', 'text'])
data.head()
