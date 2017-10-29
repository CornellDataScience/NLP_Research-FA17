# Auto Business Tagger for Yelp

## Introduction
Auto business tagging for [Yelp Dataset Challenge Round 10](https://www.yelp.com/dataset/challenge), inspired by the implementation of Yelp Machine Learning Team ([article here](https://engineeringblog.yelp.com/2015/09/automatically-categorizing-yelp-businesses.html)).

The purpose of this folder is to establish the baseline of business tag matching model performance by simulating the current implementation by Yelp.

## Implementation
According to the article, Yelp is using the multi-label classifier with ensemble method, Random Forest, by building the weak binary classifier (they used Logistic Regression).

### Binary Classifier
Following the implementation by Yelp, I used the features as follows:
```python
{
  'name' : '', # name of the business
  'last_name' : '', # last word in the name of the business
  'review' : '', # review for the business
}
```
The implementation on the article uses NAICS code and country, but I didn't include them to our feature vector since A) Yelp dataset challenge does not include NAICS, B) Yelp dataset limits the location to the major cities and the country information is not as important.

As the article suggests, I used Logistic Regression as a base classifier for the ensemble method. I tried both weighted class and without balancing, but f-1 score improved without balancing.

### Multi-label Classifier
Coming soon

### Results
Our purpose of the study is to analyze if this method is sufficient to extract subtle classifications, such as 'Dim Sum' from 'Chinese'. For this reason, I filtered all business with 'Chinese' tag. Also define the sub-topic by the network analysis tool I built ([here](../co_occurrence_net)).

I used predicting 'Dim Sum' from 'Chinese' as a target pilot run, and the weak binary classifier marked 87% accuracy with f-1 score of .76

Some strongly correlated features are:
- Word 'dim' in review (0.72)
- Word 'sum' in review (0.63)
- Business name ends with 'sum' (0.09)

Interesting founding:
- Word 'china' is negatively correlated (-0.15)
- Word 'shanghai' is positively correlated (0.086)


## Documentation
Coming soon
