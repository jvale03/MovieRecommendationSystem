import os
import numpy as np
import pandas as pd

db_path = "../DataSet/"

# read csv files from DB
user_ratings_df = pd.read_csv(db_path+"ratings.csv")
movie_metadata_df = pd.read_csv(db_path+"movies.csv")


# getting only the necessary collumns
user_ratings_df = user_ratings_df[["userId","movieId","rating"]]

# New dataframe where each column represent a user and each row represents a movie, the values are the ratings
final_df = user_ratings_df.pivot(index="movieId",columns="userId",values="rating")
final_df.fillna(0,inplace=True) # replace NaN values with 0

print(user_ratings_df.head())
print("-----------")
print(movie_metadata_df.head())
print("-----------")
print(final_df.head())