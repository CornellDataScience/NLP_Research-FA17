# LDA Topic Model Generator

The LDA topic model and distribution generation for [Yelp Dataset Challenge Round 10](https://www.yelp.com/dataset/challenge).
## Introduction

This is a wrapper class to create a k-dimensional topic distribution for one text.

## Documentation
#### Parameters:
*vectorizer(CountVectorizer)* : Count vectorizer object
*feature_names(list)* : List of vocabulary
*lda_model(LatentDirichletAllocation)* : Topic model

#### Methods:
```python
__init__(model = None, counter = None, max_features = 100000)
```
If loading pre-trained model, use model, counter parameter at the point of instantiation. If not specified, you need to train the topic model using fit function.

*display_topics(n_top_words, topic_n = None)* : Display the top n words for each topic in the model

*fit(texts, n = 100)* : Train the LDA model with given list of documents

*embed(text, method)* : Embed the review text into k-dimensional topic vector

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
