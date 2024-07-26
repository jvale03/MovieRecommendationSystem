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
tags = pd.read_csv(db_path+"tag.csv")
ratings = None
    

def process_dataset():
    global movies, tags,ratings
    ratings = pd.read_csv(db_path+"rating.csv")

    
    # remove unnecessary columns
    columns_to_rm = ["userId", "timestamp"]
    for column in columns_to_rm:
        try:
            tags = tags.drop(columns=[column])
        except:
            continue        

    tags.to_csv(db_path+"tag.csv",index=False)

    try:
        tags['tag'] = tags['tag'].astype(str).fillna('')

        # group tags by movie Id
        tags_grouped = tags.groupby("id")["tag"].apply(lambda x: ' '.join(x)).reset_index()

        # merge datasets
        movies = pd.merge(movies, tags_grouped, on='id', how='left')

        # fill Nan with empty string
        movies['tag'] = movies['tag'].fillna('')
        
        movies.to_csv(db_path+"movie.csv",index=False)
    except:
        None

    # remove users that voted less than 15 times
    user_counts = ratings['userId'].value_counts()

    valid_users = user_counts[user_counts >= 80].index

    ratings = ratings[ratings['userId'].isin(valid_users)]

    ratings.to_csv(db_path+"rating.csv",index=False)



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
    return "".join(x["genres"]) + " " + " ".join(x["tag"])



# CountVectorizer and cosine similarity algorithms
def data_vectorizer():
    global movies

    features = ["genres","tag"]

    # prepare data
    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)

    movies["soup"] = movies.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(movies["soup"])

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
    sim_scores = sim_scores[1:21]
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


