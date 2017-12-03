import os
import gensim
import multiprocessing as mp


class Corpus(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith('.txt'):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()

if __name__ == "__main__":
    train_corpus = Corpus(os.getcwd())
    params = {'size': 200, 'window': 10, 'min_count': 10, 'workers': max(1, mp.cpu_count() - 1), 'sample': 1E-3, }
    model = gensim.models.Word2Vec(train_corpus, **params)
    print(model.wv['live'])
