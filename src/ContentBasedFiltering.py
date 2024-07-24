import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
import numpy as np
from joblib import dump,load
from time import sleep


db_path = "../DataSet/"
model_path = "../Model/"

movies = pd.read_csv(db_path+"movie.csv",low_memory=False)
keywords = None
ratings = None
credits = None
    

# not necessary to our database
def process_dataset():
    global movies
    # remove unnecessary columns from csv
    columns_to_rm = ["belongs_to_collection","budget","homepage","imdb_id","original_language","original_title","adult","popularity","poster_path","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","tagline","video","vote_average","vote_count"]
    for column in columns_to_rm:
        try:
            movies = movies.drop(columns=[column])
        except:
            continue        
    movies.to_csv(db_path+"movies_metadata.csv",index=False)
    
    # convert Id column to string
    keywords["id"] = keywords["id"].astype(str)
    movies["id"] = movies["id"].astype(str)
    credits["id"] = credits["id"].astype(str)
    ratings["movieId"] = ratings["movieId"].astype(str)

    # made changes only if necessary
    if "keywords" not in movies.columns or "cast" not in movies.columns:

        # merge keywords and credits in movies.csv file
        if "keywords" not in movies.columns:
            movies = pd.merge(keywords, movies, on="id")

        if "cast" not in movies.columns and "crew" not in movies.columns:
            movies = pd.merge(movies,credits, on="id")
        
        movies.to_csv(db_path+"movies_metadata.csv",index=False)


# not necessary for this database
# extract director name from the crew feature
def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    
    return np.nan

# return the list top 3 or the entire list
def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

# convert to lower case and strips names of spaces
def clean_data(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ", "")) for i in x]
    
    else:
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            return ""

# feed cleaned data, given to vectorizer
def create_soup(x):
    return "".join(x["genres"])



# CountVectorizer and cosine similarity algorithms
def data_vectorizer():
    global movies

    # prepare data
    movies["genres"] = movies["genres"].apply(clean_data)

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(movies["genres"])

    # compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    movies = movies.reset_index()

    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    return indices, cosine_sim


# recommendation based up on credits, keywords and genres
def content_based_filtering(movie_name,indices,cosine_sim):

    # movie index that matches the title
    try:
        idx = indices[movie_name]
    except:
        return []


    # get and sort the similarity scores of all movies with the choosen one
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the 10 most similar
    sim_scores = sim_scores[1:16]
    # get the indices
    movie_indices = [i[0] for i in sim_scores]

    # convert to list
    mvs_list = movies["title"].iloc[movie_indices].tolist()
    return mvs_list

    

def save_model(model):
    dump(model,model_path+"ContentBasedFiltering.joblib")

def load_model():
    model = load(model_path + 'ContentBasedFiltering.joblib')
    return model



