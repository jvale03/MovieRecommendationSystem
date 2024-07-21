import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

db_path = "../DataSet/"

# read csv files from DB
ratings = pd.read_csv(db_path+"ratings.csv")
movies = pd.read_csv(db_path+"movies_metadata.csv")
movies["title"] = movies["title"].astype(str)


def algorithm_prepare():
    # New dataframe where each column represent a user and each row represents a movie, the values are the ratings
    final_df = ratings.pivot(index="movieId",columns="userId",values="rating")
    final_df.fillna(0,inplace=True) # replace NaN values with 0

    # get the number of movies that each user voted and number of users that voted in each movie
    nr_user_voted = ratings.groupby("movieId")["rating"].agg("count")
    nr_movies_voted = ratings.groupby("userId")["rating"].agg("count")

    # remove users that voted in less than 10 movies and movies that have less than 50 votes
    final_df = final_df.loc[nr_user_voted[nr_user_voted > 10].index,:]
    final_df = final_df.loc[:,nr_movies_voted[nr_movies_voted > 50].index]


    matrix = csr_matrix(final_df.values)
    final_df.reset_index(inplace=True)
    # print(matrix)

    knn = NearestNeighbors(n_jobs=-1, n_neighbors=20, algorithm="brute", metric="cosine")
    knn.fit(matrix)

    return knn, final_df, matrix


# Item-Based Colaborative Filtering to find the 10 movies closest to the respective one 
def collaborative_filtering(movie_name, knn, final_df, matrix):
    nr_movies_recomend = 10
    # obtain a list with movies that contain that name
    movie_list = movies[movies['title'].str.contains(movie_name)]
    movie_list = movie_list["id"]
    print(movie_list)

    if len(movie_list):
        # obtain the id of that movie and is correspondent index
        movie_idx = movie_list.iloc[0]['id']
        movie_idx = final_df[final_df['id'] == movie_idx].index[0]

        # knn algorithm search for the closest movies to the given movie, returning the distances and index of the NN
        distances , indices = knn.kneighbors(matrix[movie_idx],n_neighbors=nr_movies_recomend+1)    

        # sort the movies, removing the first one (movie_name)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        
        # list with recomended movies and their distances
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_df.iloc[val[0]]['id']
            idx = movies[movies['id'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        
        print(recommend_frame)
        # return the dataframe with recomendations
        df = pd.DataFrame(recommend_frame,index=range(1,nr_movies_recomend+1))
        return df
    else:
        return "\033[31mNo movies found!\033[m"







