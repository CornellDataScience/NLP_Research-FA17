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

class TextEmbedder(object):
    def __init__(self, **kwargs):
        self.trained = True
        self.business_idf = False
        self.user_idf = False
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

            # load user_idf if exists
            if 'business_idf' not in kwargs.keys():
                self.business_idf_dict = None
            else:
                self.business_idf_dict = kwargs['business_idf']
                self.business_idf = True


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
            tf = self.augmented_embed_sent(text, alpha, minimum_probability)
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
            tf = augmented_embed_sent(text, alpha, minimum_probability)
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
            tf = augmented_embed_sent(text, alpha, minimum_probability)
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


if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('../workspace/gensim/chinsese_dict.dict')
    model =  models.LdaModel.load('../workspace/gensim/lda.model')
    model = TextEmbedder(model = model, dictionary = dictionary)
    sample = "This review is based upon consistency of flavor and great customer service.\
      We came and there was an unknown issue that required a 25 minute wait for food.  \
      The employee notified us, and although hesitant, we decided to stay.  \
      We have been here numerous times before in the past years so we are familiar with this location.  \
      The employee was apologetic and gave us a free drink.  \
      That was a simple gesture but rarely do you see decent customer service anymore.  \
      We received our food and had an issue with an incorrect order.  \
      It was explained and the issue was resolved quickly.  They gave us a free appetizer.  \
      We do not expect perfection, nor free food.  \
      This restaurant cares for customers and works to provide a positive experience.  \
      We would return again because they have good food and they care.  \
      That is a rarity in today's restaurant culture.  Kudos to the manager for creating this culture. \
      Ordered- fried rive and Tofu, edamame, won ton soup, dynamite chx, and Thai curry."
    print (model.augmented_embed_text(sample))
