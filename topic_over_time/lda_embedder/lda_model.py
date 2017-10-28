from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class LDAembedder(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.vectorizer = kwargs['counter']
            self.feature_names = self.vectorizer.get_feature_names()
            self.lda_model = model
        else:
            if kwargs['max_features']:
                max_features = kwargs['max_features']
            else:
                max_features = 100000
            self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features = max_features)
            self.feature_names = None
            self.lda_model = None


    def display_topics(self, n_top_words, topic_n = None):
        '''
        Display the top n words for each topic in the model.

        Input:
            n_top_words(int) : the number of words to display for each topic
            (Optional)
            topic_n(int) : if specified, only diplay the topic_nth topic
        '''
        model = self.lda_model
        feature_names = self.feature_names

        if model and feature_names:
            if topic_n:
                topic = model.components_[topic_n]
                print("Topic %d:" % topic_n)
                print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
            else:
                for topic_index, topic in enumerate(model.components_):
                    print("Topic %d:" % topic_index)
                    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
        else:
            raise Exception('You have to train before display')


    def fit(self, texts, n = 100):
        '''
        Train the LDA with n number of topics

        Input:
            texts(list) : list of documents. Each item in the list is a string type
            n(int) : number of topics, if not specified n = 100
        '''
        count = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names()
        model = LatentDirichletAllocation(n_topics=n).fit(count)
        self.lda_model = model

    def embed(self, text, method = 'additive'):
        '''
        embed the review text into k-dimensional topic vector

        Input:
            text(string) : a document to embed
            method(string) : 'additive' or 'multiplicative'

        Output:
            the vector of length k (k = number of topics)
        '''
        tokenizer = self.vectorizer.build_analyzer()
        count = self.vectorizer.transform(tokenizer(text))

        dirich = self.lda_model.transform(count)
        if method == 'additive':
            total = np.array([1]*dirich[0])
            for i in dirich:
                total = np.add(total, i)
            return total / sum(total)
        elif method == 'multiplicative':
            product = np.array([1]*dirich[0])
            for i in dirich:
                product = np.multiply(product, i)
            return product / sum(product)
        else:
            return dirich
