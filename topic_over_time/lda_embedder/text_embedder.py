'''
Topic-based Text embedding class for Round 10 Yelp Dataset Challenge
'''

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
import nltk
import numpy as np
import pickle

class TextEmbedder(object):
    def __init__(self, **kwargs):
        self.trained = True
        self.business_idf = False
        self.user_idf = False
        self.business_tfidf = False
        if kwargs != None:
            # load dictionary if exists
            if 'dictionary' not in kwargs.keys():
                self.dictionary = None
                self.trained = False
            else:
                self.dictionary = kwargs['dictionary']

            # load topic model if exists
            if 'model' not in kwargs.keys():
                self.model = None
                self.trained = False
            else:
                self.model = kwargs['model']

            # load user_idf if exists
            if 'user_idf' not in kwargs.keys():
                self.user_idf_dict = None
            else:
                self.user_idf_dict = kwargs['user_idf']
                self.user_idf = True

            # load business_idf if exists
            if 'business_idf' not in kwargs.keys():
                self.business_idf_dict = None
            else:
                self.business_idf_dict = kwargs['business_idf']
                self.business_idf = True

            # load business_tfidf if exists
            if 'business_tfidf' not in kwargs.keys():
                self.business_tfidf_dict = None
            else:
                self.business_tfidf_dict = kwargs['business_tfidf']
                self.business_tfidf = True

    def tokenize_(self, text):
        '''
        tokenize the text using the utils from Gensim

        Input:
            text(str) : text that you want to tokenize
        Output
            (list): tokenized text
        '''
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]

    def embed(self, text, minimum_probability = None):
        '''
        embed the raw text into the vector with the length of topic number

        Input:
            text(str) : text that you want to embed
        Output:
            (list): embedded text
        '''
        if self.trained:
            model = self.model
            dictionary = self.dictionary
            text = self.tokenize_(text)
            bow = self.dictionary.doc2bow(text)
            kindex = self.model.get_document_topics(bow, minimum_probability)
            out = [0.] * self.model.num_topics
            for i, p in kindex:
                out[i] = p
            return np.array(out)
        else:
            print ('Load LDA model and dictionary')
            return None


    def embed_sent(self, text, minimum_probability = None):
        '''
        tokenize the text by sentences first, then
        embed the raw text into the vector with the length of topic number

        Input:
            text(str) : text that you want to embed
            minimum_probability(float) : the lowest threshold to be counted as one of topics.
            None otherwise
        Output:
            (list): embedded text
        '''
        if self.trained:
            model = self.model
            dictionary = self.dictionary
            out = np.array([0.] * self.model.num_topics)
            sentences = len(nltk.sent_tokenize(text))
            for text in nltk.sent_tokenize(text):
                out += self.embed(text, minimum_probability)
            return (out/sentences)
        else:
            print ('Load LDA model and dictionary')
            return None


    def embed_bow(self, text):
        '''
        return a sparse matrix of bag of words model

        Input:
            text(str) : text that you want to embed
        Output:
            (list): embedded text
        '''
        if self.trained:
            model = self.model
            text = self.tokenize_(text)
            bow = self.dictionary.doc2bow(text)
            return bow
        else:
            print ('Load LDA model and dictionary')
            return None


    def augmented_embed_text(self, text, alpha = 0.5, minimum_probability = 0.0):
        '''
        add scaling to normalize documents with large sentence counts
        smooth the result by alpha value, rescale the result so that the
        sum of embedding becomes 1.0

        Input:
            text(str) : text that you want to embed
            minimum_probability(float) : the lowest threshold to be counted as one of topics.
            None otherwise
        Output:
            (list): embedded text
        '''
        if self.trained:

            out = np.array([0.]*128)
            sentences = len(nltk.sent_tokenize(text))
            for text in nltk.sent_tokenize(text):
                out += self.embed(text,minimum_probability)

            out = alpha + alpha * out/max(out)

            return out/sum(out)
        else:
            print ('Load LDA model and dictionary')
            return None

    def user_tfidf_embed(self, text, user_id, alpha = 0.5, minimum_probability = 0.0):
        '''
        embed the text with tf-idf of the topic values for each user
        This will scale the embedding by penalizing frequently mentioned
        topic for one user
        '''
        # for now not allowing this function without loading idf
        if self.trained and self.user_idf:
            tf = self.augmented_embed_text(text, alpha, minimum_probability)
            idf = self.user_idf_dict[user_id]
            out = np.multiply(tf, idf)
            if sum(out) == 0.0:
                print ('User has too low idf')
                return np.array([0.]*128)
            return out/sum(out)
        else:
            print ('Load LDA model and dictionary and user idf')
            return None

    def user_tf_business_idf(self, text, business_id, alpha = 0.5, minimum_probability = 0.0):
        '''
        embed the text with tf-idf of the topic values for each business
        This will scale the embedding by penalizing frequently mentioned
        topic for each business
        '''
        # for now not allowing this function without loading idf
        if self.trained and self.business_idf:
            tf = self.augmented_embed_text(text, alpha, minimum_probability)
            idf = self.business_idf_dict[business_id]
            out = np.multiply(tf, idf)
            if sum(out) == 0.0:
                print ('Business has too low idf')
                return np.array([0.]*128)
            return out/sum(out)
        else:
            print ('Load LDA model and dictionary and user idf')
            return None

    def user_tfidf_business_idf(self, text, user_id, business_id, alpha = 0.5, minimum_probability = 0.0):
        '''
        embed the text with tf-idf of the topic values for each user, then
        multiply this value by topic idf for each business

        Give high value of the review that by the user who doesn't usually
        talk about topicA and topicA is importnant characteristic of the particular
        business
        '''
        # for now not allowing this function without loading idf
        if self.trained and self.business_idf and self.user_idf:
            tf = self.augmented_embed_text(text, alpha, minimum_probability)
            bidf = self.business_idf_dict[business_id]
            uidf = self.user_idf_dict[user_id]
            out = np.multiply(np.multiply(tf, uidf), bidf)
            if sum(out) == 0.0:
                print ('Business has too low idf')
                return np.array([0.]*128)
            return out/sum(out)
        else:
            print ('Load LDA model and dictionary and user idf')
            return None

    def augmented_tf_business_tfidf(self, text, business_id, alpha = 0.5, minimum_probability = 0.0):
        '''
        embed the text with tf-idf of the topic values for each user, then
        multiply this value by topic idf for each business

        Give high value of the review that by the user who doesn't usually
        talk about topicA and topicA is importnant characteristic of the particular
        business
        '''
        # for now not allowing this function without loading idf
        if self.trained and self.business_tfidf:
            tf = self.augmented_embed_text(text, alpha, minimum_probability)
            btfidf = self.business_tfidf_dict[business_id]
            out = np.multiply(tf, btfidf)
            if sum(out) == 0.0:
                print ('Business has too low tfidf')
                return np.array([0.]*128)
            return out/sum(out)
        else:
            print ('Load LDA model and dictionary and user idf')
            return None


if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('../data/gensim/chinsese_dict.dict')
    model =  models.LdaModel.load('../data/gensim/lda.model')

    with open('../data/u_idf.pickle', 'rb') as f:
        uidf_data = pickle.load(f)

    with open('../data/b_idf.pickle', 'rb') as f:
        bidf_data = pickle.load(f)

    with open('../data/b_tfidf.pickle', 'rb') as f:
        btfidf_data = pickle.load(f)


    model = TextEmbedder(model = model, dictionary = dictionary, user_idf = uidf_data, business_idf = bidf_data, business_tfidf = btfidf_data)

    user1 = 'CxDOIDnH8gp9KXzpBHJYXw'
    business = 'gtcsOodbmk4E0TulYHnlHA'

    sample = "Bar Crawl #1:  Beau's Kissmeyer Nordic Pale Ale & St Bernardus Abt 12\
    \n\nHappy Hour Everyday till 7 pm! $2 off drafts and bottles/cans. Sweet!\
    \n\nOf course I have to start off with a pint: Beau's Kissmeyer Nordic Pale Ale ($5.50 with HH specials!) \
    and my Yelp friend with a can of Howe Sound Lager ($4.5 with HH Special). \
    And of course if you prefer some exquisite import: A bottle of St Bernardus Abt 12 ($28).\
    \n\nThe interior is dark, dim and cozy. I like how Northwood is a coffee/beer/cocktail drinking \
    place as that means I can hang out here the whole day. \
    \n\nToo bad they were out of the 8oz Cold Brew to go. \
    I guess I have to come back soon to try out their cocktails and coffee! And free WIFI!! \
    Oh that means I can even yelp a bit!"

    print (model.augmented_embed_text(sample))
    print (model.user_tfidf_embed(sample, user1))
    print (model.user_tf_business_idf(sample, business))
    print (model.user_tfidf_business_idf(sample, user1, business))

    print (model.augmented_tf_business_tfidf(sample, business))
