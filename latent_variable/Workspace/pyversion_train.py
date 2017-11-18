import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import seaborn as sns
import json
import matplotlib.pyplot as plt
from gensim import corpora, models
import pickle

rest_review = pd.read_csv('rest_review.csv')
print (rest_review)
#preprocess
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
en_stop = stopwords.words('english')

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

def preprocess(text):
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    
    pospeech=[]
    tag = nltk.pos_tag(tokens)
    for j in tag:
        if j[1] == 'NN' or j[1] == 'JJ':
            pospeech.append(j[0])
    # remove stop words from tokens
    stopped_tokens = [i for i in pospeech if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    return stemmed_tokens

NUM_TOPICS = 120
from gensim import corpora, models


def getlda(df):
    texts = []
    for i in df['text']:
        texts.append(preprocess(i))
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    # generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes=20)
    return ldamodel

def getprefer(userid, ldamodel, df):
    user_reviews = df.loc[df['user_id'] == userid]
    l = np.zeros(NUM_TOPICS)
    texts = []
    for t in user_reviews['text']:
        texts.append(preprocess(t))
    
    for i in ldamodel.get_document_topics(corpus):
        for topic in i:
            l[topic[0]] += topic[1]
    topic_likelihood = []
    for i in l:
        topic_likelihood.append(i/sum(l))
    return topic_likelihood


from scipy.optimize import minimize
def min_loss(raw_prefer, actual_rating):
    dim = NUM_TOPICS
    prefer = []
    for i in raw_prefer:
        prefer.append(np.asarray(i, dtype=np.float32))
    prefer = np.asarray(prefer)
    print (prefer)
    print (actual_rating)
    bound = []
    for i in range(0,dim):
        bound.append((1.0,5.0))
    bnds = tuple(bound)


    def f(x,prefer,actual_rating):

        return sum(abs(np.dot(x,np.transpose(prefer)) - actual_rating))
    initial_guess = [np.random.uniform(1.0,5.0,dim)]
    #initial_guess = np.full(5,2.5)
    result = minimize(f, initial_guess, args=(prefer,actual_rating), method='SLSQP', bounds=bnds)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
        print(result.fun)
        print(result.nit)
        return result
    else:
        raise ValueError(result.message)

def add_prefer_to_df(df, lda, groupbyusers):
    d = {}
    new_df = df
    for u in groupbyusers['user_id']:
        d[u] = getprefer(u,lda,new_df)
    df['preference'] = df.apply(lambda x: d[x['user_id']],axis=1)
    return new_df

def train_rest_subscore(bizid, df):
    biz = df.loc[df['business_id'] == bizid]
    rating = biz['stars']
    preference = biz['preference']
    result = min_loss(preference.values, rating.values)
    return result.x


def add_subscore_to_df(df, groupbybiz):
    d = {}
    new_df = df
    for u in groupbybiz['business_id']:
        d[u] = train_rest_subscore(u, df)
    df['subscore'] = df.apply(lambda x: d[x['business_id']],axis=1)
    return new_df

lda = getlda(rest_review)
with open('lda.pickle', 'wb') as handle:
    pickle.dump(lda, handle, protocol=pickle.HIGHEST_PROTOCOL)

groupby_user = rest_review.groupby('user_id').size().reset_index(name='counts')
groupby_user.to_csv()

prefer_added = add_prefer_to_df(rest_review, lda, groupby_user)
prefer_added.to_csv()

subscore_added = add_subscore_to_df(prefer_added, lda, group)
subscore_added.to_csv()