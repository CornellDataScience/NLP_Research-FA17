# Local Yelp Expert Identifier 

## Introduction
The objective of the project is to build and combine two models (Local Expert Identifier / Topical Expert Identifier) for 
the purpose of identifying 'experts' among yelp users. 

### Local Expert Identifier
The "Local Expert Identifier" is a Gaussian Mixture Model that identifies clusters in a given user's review locations. The model then uses the center of the most probable mixture component to predict the user's most probable location. 

### Topical Expert Identifier
The "Topical Expert Identifier" is currently a supervised learning algorithm that combines different features about the users reviews in a certain category in order to determine if they are an expert in that category. The goal is to see if an unsupervised algorithm would be able to classify users into clusters of expert and non-expert without needing labels. 


