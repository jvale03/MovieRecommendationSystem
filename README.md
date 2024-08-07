# MovieRecommendationSystem

Machine Learning project to recommend a movie based on **movie information** and **users rating**. 

## Data Preprocessing

In this case, processing our [DataSet](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv) is as simple as merging the **tag** and **movie** CSVs, but it depends on DB used, so adapt it as necessary. 

## Project tree

```
- /MovieRecommendationSystem
    - /src
        - ...
    - /DataSet
        - movie.csv
        - rating.csv
        - tag.csv
    - /Model
        - ...
```

## Recommendation System Types

### Content Based Filtering
Recommendation strategy that suggests similar movies based on their attributes. However it limits exposure to different products,  not allowing the exploration of new content.

### Collaborative Filtering
Recommendation strategy that considers and compares others users in the database. There are various approaches to implementing collaborative filtering, but the fundamental concept is the collective influence of multiple users on the recommendation outcome.

### Content and Collaborative Filtering
As I haven't found any way to combine the two strategies and there is no correct way to do this, thinking of implementing it as follows:

Having the output of each of the strategies mentioned, weights are assigned to each of the films given their position in what is considered the optimal result. Once this is done, we combine the results by adding the weights in case there are coincident films in each of the outputs, in order to obtain a final ordered list with the 10 best recommendations.

## Steps
### Process DataSet
First time running the projects it's necessary to **process the dataset**, I recommend to use the DB mencioned in **Refs**.

### Process models
As the datasets are very large, the models were not made available and it is necessary to create them, saving them later in memory so that they can be used later without using more resources multiple times.

### Get Recommendation
Return the top 10 similar movies based on your input.


## Refs: 

1. Colaborative Filtering: [link](https://www.analyticsvidhya.com/blog/2020/11/create-your-own-movie-movie-recommendation-system/)

2. Content-Based Filtering: [link](https://medium.com/web-mining-is688-spring-2021/content-based-movie-recommendation-system-72f122641eab)

3. DataSet: [link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)
