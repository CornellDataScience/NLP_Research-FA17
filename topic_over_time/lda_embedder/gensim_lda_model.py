import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
import nltk
import numpy as np

class Gensimembedder(object):
    def __init__(self, **kwargs):
        self.trained = True
        if kwargs != None:
            if 'dictionary' not in kwargs.keys():
                self.dictionary = None
                self.trained = False
            else:
                self.dictionary = kwargs['dictionary']

            if 'model' not in kwargs.keys():
                self.model = None
                self.trained = False
            else:
                self.model = kwargs['model']

    def tokenize_(self, text):
        '''
        tokenize the text using the utils from Gensim

        Input:
            text(str) : text that you want to tokenize
        Output
            (list): tokenized text
        '''
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]

    def embed(self, text):
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
            kindex = self.model.get_document_topics(bow)
            out = [0.] * self.model.num_topics
            for i, p in kindex:
                out[i] = p
            return np.array(out)
        else:
            print ('Load LDA model and dictionary')
            return None


    def embed_sent(self, text):
        '''
        tokenize the text by sentences first, then
        embed the raw text into the vector with the length of topic number

        Input:
            text(str) : text that you want to embed
        Output:
            (list): embedded text
        '''
        if self.trained:
            model = self.model
            dictionary = self.dictionary
            out = np.array([0.] * self.model.num_topics)
            sentences = len(nltk.sent_tokenize(text))
            for text in nltk.sent_tokenize(text):
                out += self.embed(text)
            return (out/sentences)
        else:
            print ('Load LDA model and dictionary')
            return None


if __name__ == '__main__':
    dictionary = corpora.Dictionary.load('../workspace/gensim/chinsese_dict.dict')
    model =  models.LdaModel.load('../workspace/gensim/lda.model')
    model = Gensimembedder(model = model, dictionary = dictionary)
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
    print (model.embed_sent(sample))
