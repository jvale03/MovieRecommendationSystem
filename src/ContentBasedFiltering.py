import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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


# recommendation based up on credits, keywords and genres
def get_recommendarion_2():
    features = ["cast","crew","keywords","genres"]
    # transformar todos os elemetnos em objetos
    for feature in features:
        movies[feature] = movies[feature].apply(literal_eval)

    # aplicar a todos os elementos da coluna "crew" o get_director
    movies["director"] = movies["crew"].apply(get_director)

    features.remove["crew"]

    for feature in features:
        movies[feature] = movies[feature].apply(get_list)

    print(movies.head())

    features.append("director")

    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)

    movies["soup"] = movies.apply(create_soup, axis=1)

get_recommendarion_2()
    
#print(get_recommendation("Toy Story"))


