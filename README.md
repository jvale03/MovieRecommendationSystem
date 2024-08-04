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
Recommendation strategy that considers and compares others users in the database. 

### Refs: 

1. Colaborative Filtering: [link](https://www.analyticsvidhya.com/blog/2020/11/create-your-own-movie-movie-recommendation-system/)

2. Content-Based Filtering: [link](https://medium.com/web-mining-is688-spring-2021/content-based-movie-recommendation-system-72f122641eab)

3. DataSet: [link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)
