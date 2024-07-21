import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
import numpy as np


db_path = "../DataSet/"

# csvs
keywords = pd.read_csv(db_path+"keywords.csv",low_memory=False)
movies = pd.read_csv(db_path+"movies_metadata.csv",low_memory=False)
credits = pd.read_csv(db_path+"credits.csv",low_memory=False)
ratings = pd.read_csv(db_path+"ratings.csv",low_memory=False)

    

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
    return " ".join(x["keywords"]) + " " + " ".join(x["cast"]) + " " + x["director"] + " " + " ".join(x["genres"])


# prepare data to be used in our vectorizers
def data_prepare():
    global movies

    features = ["cast","crew","keywords","genres"]
    # transformar todos os elemetnos em objetos
    for feature in features:
        movies[feature] = movies[feature].apply(literal_eval)

    # aplicar a todos os elementos da coluna "crew" o get_director
    movies["director"] = movies["crew"].apply(get_director)

    features = ["cast","keywords","genres"]

    for feature in features:
        movies[feature] = movies[feature].apply(get_list)

    # print(movies[["title","cast","keywords","genres"]].head())

    features = ["cast","keywords","director","genres"]

    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)

    movies["soup"] = movies.apply(create_soup, axis=1)

    # print(movies[["soup"]].head())


# CountVectorizer and cosine similarity algorithms
def data_vectorizer():
    global movies

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(movies["soup"])

    # compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    movies = movies.reset_index()

    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    return indices, cosine_sim


# recommendation based up on credits, keywords and genres
def get_recommendation(movie_name):
    indices, cosine_sim = data_vectorizer()

    # movie index that matches the title
    try:
        idx = indices[movie_name]
    except:
        return "\033[31mNo movies found!\033[m"


    # get and sort the similarity scores of all movies with the choosen one
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the 10 most similar
    sim_scores = sim_scores[1:11]
    # get the indices
    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices]
    

data_prepare()
print(get_recommendation("Catwalk"))


