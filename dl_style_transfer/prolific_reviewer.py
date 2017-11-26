import nltk
import pandas as pd
import numpy as np
import json
import os

frame = pd.read_json("/home/cai29/merged.json", lines=True)
a = frame['sum(text)'].idxmax()
print(frame['user_id'][a])


# largest = -1
# path = "/home/cai29/user_text_sentences.json"
# for filename in os.listdir(path):
#     if filename.endswith(".json"):
#         print(path + "/" + filename)
#         file = open(path + '/' + filename)
#         j = json.load(file)
#         frame = pd.DataFrame(j, columns=['user_id', 'sum(text)'])
#         if largest == -1:
#             largest = frame[frame['sum(text)'].argmax(), :]
#         else:
#             a = frame[frame['sum(text)'].argmax(), :]
#             if a['sum(text)'] > largest['sum(text)']:
#                 largest = a
#
# print(largest)
