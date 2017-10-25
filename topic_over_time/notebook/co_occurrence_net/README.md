# Business Tag Cooccurrence Network

## Introduction
Business tag co-occurrence analysis for [Yelp Dataset Challenge Round 10](https://www.yelp.com/dataset/challenge).

<p align = 'center'>
![tag net](https://camo.githubusercontent.com/d7b97d7c0873e949f827918763174efcca6c4a5f/687474703a2f2f64336a732e6f72672f65782f666f7263652e706e67)
</p>

This is a wrapper class to help analyze the co-occurrence of business category tags to determine hierarchical relationship between tags.

CategoryMap object observes the sequence of lists, such as category columns in yelp dataset. For every new tag, CategoryMap object instantiates CategoryNode object, which contains category name and number of appearance. CategoryNode object also contains cooccurrence attribute, which maps the name of category to the number of cooccurrence.

For example, if we observe a sequence of business tags such that
```python
['Restaurants', 'Chinese']
['Restaurants', 'Chinese', 'Fast Food']
['Restaurants', 'Italian']
```
CategoryMap object will create 4 CategoryNode objects as follows:

```python
Node1 = {'name' : 'Restaurants', 'counter' : 3, 'cooccurrence' : {'Chinese':2, 'Fast Food':1, 'Italian':1}}
# node 1 is for Restaurants tag. Since we observed 'Restaurants' tag 3 times, counter gets 3. The tag 'Chinese' appeared with 'Restaurants' twice, and the cooccurrence maps 'Chinese' to 2.
Node2 = {'name' : 'Chinese', 'counter' : 2, 'cooccurrence' : {'Restaurants':2, 'Fast Food':1}}  
Node3 = {'name' : 'Fast Food', 'counter' : 1, 'cooccurrence' : {'Restaurants':1,'Chinese':1}}
Node4 = {'name' : 'Italian', 'counter' : 1, 'cooccurrence' : {'Restaurants':1}}
```

## Documentation
#### CategoryNode class
##### Parameters:
*c (str)* : The name of this object  
*counter (int)* : The number of times the name above appeared in a sequence  
*cooccurrence (dict)* : The dictionary that maps the co-occurred names to the frequency of the co-occurrence  
##### Methods:
*add_coocurence(clist)* : Add clist to the cooccurrence dictionary

#### CategoryMap class
##### Parameters:
*categories (dict)* : The dictionary that maps the name of item in the network to the Node object

##### Methods:

*build_graph(seq)* : Build a graph from the list of sequence
*observe(clist)* : Add 1 sequence to the network
*display_top_n(n)* : print n nodes in a decreasing order of appearance
*get_subcategories(c)* : get the list of sub categories for c
*shared_categories(c1, c2, sub = False)* : get the list of shared categories between c1 and c2. if sub = True, then only returns subcategories

## Example
```python
import json
import pandas as pd

# load file
data = []
with open('business.json') as data_file:
    for f in data_file:
        data.append(json.loads(f))
df = pd.DataFrame(data)

# instantiate CategoryMap object
G = CategoryMap()
# df.categories is a list of string list
G.build_graph(df.categories)
# print top 10 most observed tags
G.display_top_n(10)
# get the list of sub categories for 'Chinese' tag
sub_chinese = G.get_subcategories('Chinese')
# get the list of shared sub categories between 'Italian' and 'Chinese'
shared_sub = G.shared_categories('Chinese', 'Italian', sub = True)

```
