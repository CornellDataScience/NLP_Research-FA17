# CDS: NLP Research Team

<p align="center">
  <img src="img/image.png" style="margin:0 auto;" width="75%">
</p>


**Team Lead:** Kenta Takatsu (CS '19)  
**Advisor:** Prof. Thorsten Joachims

## About Us
We are a student-led research team from [Cornell Data Science ](https://datascience.engineering.cornell.edu/index.html)(CDS), working on Natural Language Processing projects under [Prof. Thorsten Joachims](http://www.cs.cornell.edu/people/tj/). This semester, we are participating in the [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge) to provide analytic insights from raw review texts. Our final products are research papers which makes use of machine learning algorithms and statistical validations. You can visit the subteam sections to see our individual work.

## Achivements

This past semester, we had a wide range of research topics, from recommendation system to deep style transfer. In general, we took the approach called Natural Language Processing -- an interaction between machine learning and text analysis.

All researches demonstrated remarkable results; an implementation of recommendation system that beats industry standard algorithm, an accurate analytic tool to assess business trends, a classifier to identify locally popular users, and a writing style transfer with deep learning.

## Subteams

* ### [**Extracting Rating Dimensions with Text Reviews**](/latent_variable)

   **Members:** Xuwen Shen (STAT '18), Xinzhe Yang (CS '20)   
   In order to give insights to overall ratings and then create a new personalized recommendation system based on the rating that account for his or her preferences, we were hoping to extract hidden information in reviews including an individual user’s preference and a business’s properties (scores for each feature of the business). We used LDA to get topics from the review. If we have k topics in reviews then we extract k (the same number as the number of topics in Yelp reviews) dimensions in rating for each topic respectively and then use the k dimensional rating to compute the recommendation score for an individual. Finally, we created a model combining the topics and overall ratings to get a personalized ratings for a specific user.

* ### [**Topic Modeling as a Trend-Aware Performance Metric**](/topic_over_time)     
  **Members:** Kenta Takatsu (CS '19), Caroline Chang (CS '20)   
  We are developing a stream-lined star-prediction system to better assess business performance using different types of classifiers, which accounts for the temporal trends in user review topics and the strength/weakness of business characteristics in latent space.

* ### [**Local Experts in Yelp**](/local-experts)   
  **Members:** Brandon Kates (BTRY '19), Brian Cheang (CS '20)     
  The objective of the project is to build and combine two models (Local Expert Identifier / Topical Expert Identifier) for the purpose of identifying 'experts' among yelp users.  

* ### [**Neural Style Transfer For Text**](/dl_style_transfer)   
  **Members:** Luca Leeser (INFO '18), Yuji Akimoto (ORIE '19), Ryan Butler (CS '19), Cameron Ibrahim (ORIE '20)   
  We are seeking to modify the neural style transfer algorithm proposed by [Gatys et. al.](https://arxiv.org/abs/1508.06576) to make it applicable to text. Our goal is to devise an algorithm that is able to transfer the writing style of one review onto the content of another.   

## Final Submissions
You can visit our final papers from the following links:
* [Extracting Rating Dimensions from Hidden Topics in Text Reviews](/latent_variable/submission/extracting-rating-dimensions.pdf)
* [Topic Modeling as a Trend-Aware Performance Metric](/topic_over_time/submission/CDS_final_submission.pdf)
* [Identifying Experts in the Yelp Dataset](/local-experts/submission/Final_Paper.pdf)
* [On the Use of K-Competitive Networks for Writing Style Transfer](/dl_style_transfer/submission/k-competitive-networks.pdf)

## How to get the code
The code uses git submodules, so to properly intialize those you need the `--recurse-submodules` option. Additionally, using `--depth 1` will avoid cloning the history, making the clone faster.
```
git clone --recurse-submodules --depth 1 https://github.com/CornellDataScience/Yelp-FA17.git
```
