import nltk
import pandas as pd
import numpy as np
import json
import os
import mysql.connector as base

# frame = pd.read_json("/home/cai29/merged.json", lines=True)
# a = frame['sum(text)'].idxmax()

cnct = base.connect(user='cds', password='itsalldatababy', database='yelp_db')
c = cnct.cursor()
c.execute("SELECT COUNT(*) FROM review GROUP BY user_id HAVING COUNT(*) > 200")
print(c.fetchall())
c.execute("SELECT user_id, GROUP_CONCAT(text SEPARATOR '\n') FROM review GROUP BY user_id HAVING COUNT(*) > 200;")
data = pd.DataFrame(c.fetchall(), columns=['user_id', 'text'])
m = data['text'].apply(lambda x: len(nltk.sent_tokenize(x))).idxmax()
sen = nltk.sent_tokenize(data['text'][m])
file = open("./yelp_sentences.txt", 'w')
for l in sen:
    file.write(l + "\n")


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
