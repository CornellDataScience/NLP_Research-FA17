# LDA Topic Model Generator

The LDA topic model and distribution generation for [Yelp Dataset Challenge Round 10](https://www.yelp.com/dataset/challenge).
## Introduction

This is a wrapper class to create a k-dimensional topic distribution for one text.

## Documentation

### LDAembedder
#### Parameters:
- *vectorizer(CountVectorizer)* : Count vectorizer object

- *feature_names(list)* : List of vocabulary

- *lda_model(LatentDirichletAllocation)* : Topic model

#### Methods:
```python
__init__(model = None, counter = None, max_features = 100000)
```
If loading pre-trained model, use model, counter parameter at the point of instantiation. If not specified, you need to train the topic model using fit function.

- *display_topics(n_top_words, topic_n = None)* : Display the top n words for each topic in the model

- *fit(texts, n = 100)* : Train the LDA model with given list of documents

- *embed(text, method)* : Embed the review text into k-dimensional topic vector

## Example
```python
import pickle

# load the pre-trained model
with open('vectorizer_file_name', "rb") as f:
    vectorizer = pickle.load(f)
with open('model_file_name', "rb") as f:
    model = pickle.load(f)

lda = LDAembedder(model = model, counter = vectorizer)

# print top 10 words from each topic
lda.display_topics(10)

# embed text
sample = 'This place is horrible, we were so excited to try it since I got a gift card for my birthday. We went in an ordered are whole meal and they did not except are gift card, because their system was down. Unacceptable, this would have been so helpful if we would have known this prior!!'

add = lda.embed(sample, 'additive')
prod = lda.embed(sample, 'multiplicative')

```

### Gensimembedder
#### Parameters:
- *dictionary(gensim.corpora.dictionary.Dictionary)* : Dictionary for the documents

- *model(gensim.models.ldamodel.LdaModel)* : Topic model

#### Methods:
```python
__init__(model = None, dictionary = None)
```
For this class, I didn't allow any process without loadig pre-trained models.

- *embed(text)* : Embed the review text into k-dimensional topic vector

- *embed_sent(text)* : Embed the review text into k-dimensional topic vector after tokeninze by sentences


## Example
```Python
from gensim import corpora, models

dictionary = corpora.Dictionary.load('directory/to/your/dictionary')
model =  models.LdaModel.load('directory/to/your/model')

lda = Gensimembedder(model = model, dictionary = dictionary)

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
  
print (lda.embed_sent(sample))

```

