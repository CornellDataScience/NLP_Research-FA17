from lxml import html
import requests
import pandas as pd
import nltk
import tensorflow as tf
import numpy as np


def book(title, author, link):
    return {
        'title': title,
        'author': author,
        'link': link
    }

books = [
    book("The Tragedie of Hamlet",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/2265/pg2265.html"),
    book("THE TRAGEDY OF JULIUS CAESAR",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/1785/pg1785.html"),
    book("THE TRAGEDY OF OTHELLO, MOOR OF VENICE",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/1793/pg1793.html"),
    book("Tragedie of Romeo and Juliet",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/2261/pg2261.html"),
    book("All's Well, that Ends Well",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/2246/pg2246.html"),
    book("As You Like It",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/1786/pg1786.html"),
    book("The Merchant of Venice",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/1779/pg1779.html"),
    book("The Tragedy of Macbeth",
         "William Shakespeare",
         "http://www.gutenberg.org/cache/epub/1795/pg1795.html")
]
# books = [
#     book("THE TEN BOOKS ON ARCHITECTURE",
#          "Vitruvius",
#          "https://www.gutenberg.org/files/20239/20239-h/20239-h.htm"),
#     book("THE STORY OF THE LIVING MACHINE",
#          "H.W. CONN",
#          "https://www.gutenberg.org/files/16487/16487-h/16487-h.htm"),
#     book("THE WORKS OF EDGAR ALLAN POE",
#          "Edgar Allan Poe",
#          "https://www.gutenberg.org/files/2147/2147-h/2147-h.htm"),
#     book("A TALE OF TWO CITIES",
#          "Charles Dickens",
#          "https://www.gutenberg.org/files/98/98-h/98-h.htm"),
#     book("The Tragedie of Hamlet",
#          "William Shakespeare",
#          "http://www.gutenberg.org/cache/epub/2265/pg2265.html"),
#     book("ANDERSEN’S FAIRY TALES",
#          "Hans Christian Anderson",
#          "https://www.gutenberg.org/files/1597/1597-h/1597-h.htm"),
#     book("Stories the Iroquois Tell Their Children",
#          "Mabel Powers",
#          "https://www.gutenberg.org/files/22096/22096-h/22096-h.htm"),
#     book("Frankenstein",
#          "Mary Wollstonecraft",
#          "https://www.gutenberg.org/files/84/84-h/84-h.htm"),
#     book("APPLETONS' POPULAR SCIENCE MONTHLY",
#          "WILLIAM JAY YOUMANS",
#          "https://www.gutenberg.org/files/43391/43391-h/43391-h.htm"),
#     book("THE BRIDE OF THE NILE",
#          "Georg Ebers",
#          "https://www.gutenberg.org/files/5529/5529-h/5529-h.htm"),
#     book("Folklore of Scottish Lochs and Springs",
#          "James M. Mackinlay",
#          "https://www.gutenberg.org/files/56034/56034-h/56034-h.htm"),
#     book("THE TRAGEDY OF JULIUS CAESAR",
#          "William Shakespeare",
#          "http://www.gutenberg.org/cache/epub/1785/pg1785.html"),
#     book("THE TRAGEDY OF OTHELLO, MOOR OF VENICE",
#          "William Shakespeare",
#          "http://www.gutenberg.org/cache/epub/1793/pg1793.html"),
#     book("Tragedie of Romeo and Juliet",
#          "William Shakespeare",
#          "http://www.gutenberg.org/cache/epub/2261/pg2261.html")
# ]

frames = []

for b in books:
    print(b['title'])
    page = requests.get(b['link'])
    tree = html.fromstring(page.content)

    pars = tree.xpath('//p/text()')
    b['text'] = nltk.sent_tokenize("\n".join(pars).replace("\r", ""))[10:-10]
    frame = pd.DataFrame(b)
    print(frame.shape)
    frames.append(frame)

file = open('yelp_sentences')
lines = list(file)
yelp = pd.DataFrame({'title': 'Yelp Text', 'author': 'Yelp User', 'text': lines})
frames.append(yelp)

frame = pd.concat(frames)


frame.reset_index(inplace=True)
frame['len'] = frame['text'].str.len()
print("Total: ", frame.shape)

mod = tf.keras.preprocessing.text.Tokenizer()
mod.fit_on_texts(frame['text'])
inv = dict((v,k) for k, v in mod.word_index.items())
inv[-1] = ""

def get_corpus():
    '''
    Returns the dataframe of sentence samples:

    Columns: ['index', 'author', 'link', 'text', 'title']
    '''
    return frame

def get_vocab():
    '''
    Returns a dict with learned words as keys and corresponding indices as values
    '''
    return mod.word_index

def sentence_vecs(mode="count"):
    '''
    Returns a matrix of vector representations of each sentence in the corpus. Mode can be used
    to specify what information is contained int the vector

    mode: The information contained in each sentence vector. Default is count i.e. bag of words.
        Other options are binary, freq, tfidf
    '''
    return mod.texts_to_matrix(frame['text'], mode)


def sentence_mat():
    '''
    Returns integer sequences representing each sentence in the corpus. These sequences are padded with
    -1 to the length of the longest sequence. Each integer in the sequence is the index corresponding to a
    particular word.
    '''
    seqs=mod.texts_to_sequences(frame['text'])
    # seqs = list(filter(lambda x: len(x) <= 64, seqs))
    return tf.keras.preprocessing.sequence.pad_sequences(
        seqs,
        maxlen=max(map(len, seqs)),
        value=-1,
        padding="post"
    )

def get_one_hot():
    '''
    Returns matrices corresponding to the one-hot encoding representations of sentences. These are padded
    with empty vectors to the length of the longest sentence.
    '''
    return tf.one_hot(sentence_mat(), len(mod.word_index))

def one_hot_to_word(one_hot):
    '''
    Returns the word corresponding to a given one-hot encoded vector
    '''
    return inv[np.argmax(one_hot)]


def seq_list_to_word(lis):
    '''
    Given a list containing vocab indexes, return the corresponding sentence
    '''
    return " ".join([inv[i] for i in lis])


if __name__ == '__main__':
    frame.to_json("./shake_sentences.json", orient="records")
    # file = open("./shake_sentences.txt", 'w')
    # for l in frame['text']:
    #     file.write(l + "\n")
